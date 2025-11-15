"""FastAPI backend for streaming Muse headset metrics."""

from __future__ import annotations

import asyncio
import os
import time
from typing import Any, Dict, List, Optional

import numpy as np
from fastapi import FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from pylsl import StreamInlet, resolve_byprop

import utils

BAND_NAMES = ["delta", "theta", "alpha", "beta"]


class MuseMetricsSession:
    """Connects to the Muse EEG LSL stream and emits neurofeedback metrics."""

    def __init__(
        self,
        buffer_length: int = 5,
        epoch_length: int = 1,
        overlap_length: float = 0.8,
        index_channels: Optional[List[int]] = None,
        pull_timeout: float = 1.0,
        max_pull_retries: int = 5,
        enable_mock: Optional[bool] = None,
    ) -> None:
        self.buffer_length = buffer_length
        self.epoch_length = epoch_length
        self.overlap_length = overlap_length
        self.shift_length = self.epoch_length - self.overlap_length
        self.index_channels = index_channels or [0]
        self.pull_timeout = pull_timeout
        self.max_pull_retries = max_pull_retries

        self.inlet: Optional[StreamInlet] = None
        self.fs: Optional[int] = None
        self.eeg_buffer: Optional[np.ndarray] = None
        self.band_buffer: Optional[np.ndarray] = None
        self.filter_state: Any = None

        env_mock = os.environ.get('MUSE_MOCK_MODE') or os.environ.get('MUSE_ENABLE_MOCK')
        if enable_mock is not None:
            self.allow_mock = enable_mock
        elif env_mock is not None:
            self.allow_mock = (env_mock or '').lower() in {'1', 'true', 'yes'}
        else:
            self.allow_mock = True

        self.mock_active = False
        self._mock_phase = 0.0
        self.mode = 'init'

    def _connect(self) -> None:
        if self.inlet is not None or self.mock_active:
            return

        streams = resolve_byprop('type', 'EEG', timeout=2)
        if not streams:
            if self.allow_mock:
                self._activate_mock()
                return
            raise RuntimeError("Can't find EEG stream. Ensure the Muse is streaming over LSL.")

        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        self.inlet.time_correction()

        info = self.inlet.info()
        self.fs = int(info.nominal_srate() or 256)
        self._initialize_buffers()
        self.mode = 'live'

    def _initialize_buffers(self) -> None:
        assert self.fs is not None
        n_channels = len(self.index_channels)
        self.eeg_buffer = np.zeros((int(self.fs * self.buffer_length), n_channels))
        self.filter_state = None
        n_win_test = int(
            np.floor((self.buffer_length - self.epoch_length) / self.shift_length + 1)
        )
        self.band_buffer = np.zeros((n_win_test, 4))

    def _activate_mock(self) -> None:
        self.mock_active = True
        self.fs = 256
        self._mock_phase = 0.0
        self._initialize_buffers()
        self.mode = 'mock'

    def close(self) -> None:
        if self.inlet is not None:
            try:
                self.inlet.close_stream()
            finally:
                self.inlet = None
        self.mock_active = False
        self.mode = 'init'

    def _pull_chunk(self) -> np.ndarray:
        assert self.fs is not None

        if self.mock_active:
            return self._generate_mock_chunk()

        assert self.inlet is not None

        for _ in range(self.max_pull_retries):
            chunk, _ = self.inlet.pull_chunk(
                timeout=self.pull_timeout,
                max_samples=int(self.shift_length * self.fs),
            )
            if chunk:
                return np.asarray(chunk)
        raise RuntimeError('Timed out while waiting for EEG data from the Muse headset.')

    def _generate_mock_chunk(self) -> np.ndarray:
        assert self.fs is not None
        samples = max(1, int(self.shift_length * self.fs))
        start = self._mock_phase
        t = start + (np.arange(samples) / self.fs)
        self._mock_phase += samples / self.fs

        max_index = max(self.index_channels) if self.index_channels else 0
        n_channels = max_index + 1
        chunk = np.zeros((samples, n_channels))

        base_delta = np.sin(2 * np.pi * 2 * t)
        base_theta = np.sin(2 * np.pi * 6 * t + 0.5)
        base_alpha = np.sin(2 * np.pi * 10 * t + 1.1)
        base_beta = np.sin(2 * np.pi * 18 * t + 0.25)
        composite = 35 * base_delta + 25 * base_theta + 15 * base_alpha + 10 * base_beta

        for idx in range(n_channels):
            phase = idx * 0.15
            chunk[:, idx] = composite + 5 * np.sin(2 * np.pi * 0.5 * (t + phase))

        noise = np.random.normal(0, 2, size=chunk.shape)
        chunk += noise
        return chunk

    def get_metrics(self) -> Dict[str, Any]:
        self._connect()
        assert self.fs is not None
        assert self.eeg_buffer is not None
        assert self.band_buffer is not None

        chunk = self._pull_chunk()
        ch_data = chunk[:, self.index_channels]
        if ch_data.ndim == 1:
            ch_data = ch_data.reshape(-1, len(self.index_channels))

        self.eeg_buffer, self.filter_state = utils.update_buffer(
            self.eeg_buffer, ch_data, notch=True, filter_state=self.filter_state
        )

        data_epoch = utils.get_last_data(self.eeg_buffer, int(self.epoch_length * self.fs))
        band_powers = utils.compute_band_powers(data_epoch, self.fs)
        self.band_buffer, _ = utils.update_buffer(self.band_buffer, np.asarray([band_powers]))
        smooth_band_powers = np.mean(self.band_buffer, axis=0)

        return self._format_payload(band_powers, smooth_band_powers)

    async def get_metrics_async(self) -> Dict[str, Any]:
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(None, self.get_metrics)

    def _format_payload(self, band_powers: np.ndarray, smooth_band_powers: np.ndarray) -> Dict[str, Any]:
        assert self.fs is not None
        timestamp = time.time()
        band_dict = {
            name: float(band_powers[idx])
            for idx, name in enumerate(BAND_NAMES)
        }
        smooth_band_dict = {
            name: float(smooth_band_powers[idx])
            for idx, name in enumerate(BAND_NAMES)
        }

        def safe_ratio(num: float, denom: float) -> float:
            if denom is None or abs(denom) < 1e-9:
                return 0.0
            return float(num / denom)

        alpha_relaxation = safe_ratio(smooth_band_powers[2], smooth_band_powers[0])
        beta_concentration = safe_ratio(smooth_band_powers[3], smooth_band_powers[1])
        theta_relaxation = safe_ratio(smooth_band_powers[1], smooth_band_powers[2])

        metrics = {
            'alpha_relaxation': alpha_relaxation,
            'beta_concentration': beta_concentration,
            'theta_relaxation': theta_relaxation,
        }

        return {
            'timestamp': timestamp,
            'sampling_rate': self.fs,
            'bands': band_dict,
            'smoothed_bands': smooth_band_dict,
            'metrics': metrics,
            'mode': self.mode,
        }


app = FastAPI(title='Muse Metrics API', version='0.1.0')


@app.get('/health')
async def health() -> Dict[str, str]:
    return {'status': 'ok'}


@app.get('/metrics')
async def read_metrics() -> Dict[str, Any]:
    session = MuseMetricsSession()
    try:
        return await session.get_metrics_async()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    finally:
        session.close()


@app.websocket('/ws/metrics')
async def stream_metrics(websocket: WebSocket) -> None:
    await websocket.accept()
    session = MuseMetricsSession()
    try:
        while True:
            metrics = await session.get_metrics_async()
            await websocket.send_json(metrics)
    except WebSocketDisconnect:
        pass
    except RuntimeError as exc:
        await websocket.send_json({'error': str(exc)})
    finally:
        session.close()
