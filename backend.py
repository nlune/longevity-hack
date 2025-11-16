"""FastAPI backend for streaming Muse headset metrics."""

from __future__ import annotations

import asyncio
import math
import time
import warnings
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
from fastapi import BackgroundTasks, FastAPI, HTTPException, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
import neurokit2 as nk
from neurokit2.misc import NeuroKitWarning
from pydantic import BaseModel, Field
from pylsl import StreamInlet, resolve_byprop
import json
import subprocess

import utils

BAND_NAMES = ["delta", "theta", "alpha", "beta"]
METRIC_NAMES = ["alpha_relaxation", "beta_concentration", "theta_relaxation"]
HRV_METRIC_NAMES = ["hrv_rmssd", "hrv_sdnn", "hrv_lf_hf"]
DEFAULT_BASELINE_SECONDS = 60
NEGATIVE_ALERT_METRICS = {"hrv_rmssd", "hrv_sdnn"}
POSITIVE_ALERT_METRICS = set(METRIC_NAMES + ["hrv_lf_hf"]) - NEGATIVE_ALERT_METRICS
COMPOSITE_WEIGHTS = {
    'alpha_relaxation': 1.0,
    'theta_relaxation': 0.8,
    'beta_concentration': 1.0,
    'hrv_rmssd': 1.2,
    'hrv_sdnn': 1.0,
    'hrv_lf_hf': 0.8,
}


class MetricStats(BaseModel):
    mean: float
    std: float


class BaselineRequest(BaseModel):
    duration_seconds: float = Field(DEFAULT_BASELINE_SECONDS, ge=30, le=600)
    hrv_window_seconds: float = Field(30.0, ge=10.0, le=120.0)
    hrv_step_seconds: float = Field(10.0, ge=5.0, le=60.0)


class BaselineResponse(BaseModel):
    duration_seconds: float
    sample_count: int
    hrv_window_seconds: float
    hrv_window_count: int
    metrics: Dict[str, MetricStats]


class MonitorRequest(BaseModel):
    baseline: BaselineResponse
    duration_seconds: float = Field(30.0, ge=10.0, le=180.0)
    deviation_threshold: float = Field(2.0, ge=0.5, le=6.0)


class MonitorResponse(BaseModel):
    timestamp: float
    metrics: Dict[str, float]
    deviations: Dict[str, float]
    alerts: List[str]
    sample_count: int
    hrv_sample_seconds: float
    composite_score: float


class NotificationRequest(BaseModel):
    message: str
    title: str = 'Stress Compass'
    delay_seconds: float = Field(0.0, ge=0.0, le=3600.0)


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

    def _connect(self) -> None:
        if self.inlet is not None:
            return

        streams = resolve_byprop('type', 'EEG', timeout=2)
        if not streams:
            raise RuntimeError("Can't find EEG stream. Ensure the Muse is streaming over LSL.")

        self.inlet = StreamInlet(streams[0], max_chunklen=12)
        self.inlet.time_correction()

        info = self.inlet.info()
        self.fs = int(info.nominal_srate() or 256)
        n_channels = len(self.index_channels)
        self.eeg_buffer = np.zeros((int(self.fs * self.buffer_length), n_channels))
        self.filter_state = None

        n_win_test = int(np.floor((self.buffer_length - self.epoch_length) / self.shift_length + 1))
        self.band_buffer = np.zeros((n_win_test, 4))

    def close(self) -> None:
        if self.inlet is not None:
            try:
                self.inlet.close_stream()
            finally:
                self.inlet = None

    def _pull_chunk(self) -> np.ndarray:
        assert self.inlet is not None
        assert self.fs is not None

        for _ in range(self.max_pull_retries):
            chunk, _ = self.inlet.pull_chunk(
                timeout=self.pull_timeout,
                max_samples=int(self.shift_length * self.fs),
            )
            if chunk:
                return np.asarray(chunk)
        raise RuntimeError('Timed out while waiting for EEG data from the Muse headset.')

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
        }


class PPGStreamSession:
    """Simple helper to pull PPG samples for HRV analysis."""

    def __init__(self, channel_index: int = 0, pull_timeout: float = 0.5) -> None:
        self.channel_index = channel_index
        self.pull_timeout = pull_timeout
        self.inlet: Optional[StreamInlet] = None
        self.fs: Optional[int] = None

    def _connect(self) -> None:
        if self.inlet is not None:
            return

        streams = resolve_byprop('type', 'PPG', timeout=5)
        if not streams:
            raise RuntimeError("Can't find PPG stream. Ensure the Muse PPG is streaming over LSL.")

        self.inlet = StreamInlet(streams[0], max_chunklen=64)
        self.inlet.time_correction()

        info = self.inlet.info()
        self.fs = int(info.nominal_srate() or 64)

    def close(self) -> None:
        if self.inlet is not None:
            try:
                self.inlet.close_stream()
            finally:
                self.inlet = None

    def collect_samples(self, duration_seconds: float) -> Tuple[np.ndarray, int]:
        self._connect()
        assert self.inlet is not None
        assert self.fs is not None

        samples: List[float] = []
        start_time = time.time()
        try:
            while time.time() - start_time < duration_seconds:
                chunk, _ = self.inlet.pull_chunk(timeout=self.pull_timeout)
                if not chunk:
                    continue
                arr = np.asarray(chunk)
                if arr.size == 0:
                    continue
                if arr.ndim == 1:
                    values = arr
                else:
                    idx = min(self.channel_index, arr.shape[1] - 1)
                    values = arr[:, idx]
                samples.extend(values.tolist())
        finally:
            self.close()

        return np.asarray(samples, dtype=float), self.fs


def _collect_eeg_payloads(duration_seconds: float) -> List[Dict[str, Any]]:
    session = MuseMetricsSession()
    payloads: List[Dict[str, Any]] = []
    start_time = time.time()
    try:
        while time.time() - start_time < duration_seconds:
            payloads.append(session.get_metrics())
    finally:
        session.close()
    return payloads


def _collect_ppg_samples(duration_seconds: float) -> Tuple[np.ndarray, int]:
    session = PPGStreamSession()
    return session.collect_samples(duration_seconds)


def _metric_stats(values: Sequence[float]) -> MetricStats:
    clean_values = [float(v) for v in values if v is not None and not np.isnan(v)]
    if not clean_values:
        return MetricStats(mean=0.0, std=0.0)
    arr = np.asarray(clean_values, dtype=float)
    std = float(np.std(arr, ddof=1)) if arr.size > 1 else 0.0
    return MetricStats(mean=float(np.mean(arr)), std=std)


def summarize_metric_series(metric_series: Sequence[Dict[str, float]]) -> Dict[str, MetricStats]:
    summary: Dict[str, MetricStats] = {}
    for metric in METRIC_NAMES:
        values = [row.get(metric) for row in metric_series if row.get(metric) is not None]
        summary[metric] = _metric_stats(values)
    return summary


def average_metric_series(metric_series: Sequence[Dict[str, float]]) -> Dict[str, float]:
    averages: Dict[str, float] = {}
    for metric in METRIC_NAMES:
        values = [row.get(metric) for row in metric_series if row.get(metric) is not None]
        averages[metric] = float(np.mean(values)) if values else 0.0
    return averages


def _extract_hrv_value(hrv_frame: Any, column: str) -> float:
    if hrv_frame is None or column not in hrv_frame:
        return 0.0
    series = hrv_frame[column]
    if getattr(series, 'size', 0) == 0:
        return 0.0
    value = float(series.iloc[0]) if hasattr(series, 'iloc') else float(series[0])
    if np.isnan(value):
        return 0.0
    return value


def compute_hrv_metrics(samples: np.ndarray, fs: int) -> Dict[str, float]:
    if samples.size == 0 or fs is None or fs <= 0:
        raise RuntimeError('Not enough PPG samples to compute HRV metrics.')

    signals, info = nk.ppg_process(samples, sampling_rate=fs)
    peaks = info.get('PPG_Peaks')
    if peaks is None or len(peaks) < 3:
        raise RuntimeError('Unable to detect enough PPG peaks for HRV analysis.')

    with warnings.catch_warnings():
        warnings.filterwarnings('ignore', category=NeuroKitWarning)
        hrv_time = nk.hrv(peaks, sampling_rate=fs, show=False, method='time')
        try:
            hrv_frequency = nk.hrv(peaks, sampling_rate=fs, show=False, method='frequency')
        except Exception:
            hrv_frequency = None
    return {
        'hrv_rmssd': _extract_hrv_value(hrv_time, 'HRV_RMSSD'),
        'hrv_sdnn': _extract_hrv_value(hrv_time, 'HRV_SDNN'),
        'hrv_lf_hf': _extract_hrv_value(hrv_frequency, 'HRV_LFHF'),
    }


def compute_hrv_windows(
    samples: np.ndarray,
    fs: int,
    window_seconds: float,
    step_seconds: float,
) -> List[Dict[str, float]]:
    if samples.size == 0:
        raise RuntimeError('No PPG samples were collected for HRV computation.')
    if fs is None or fs <= 0:
        raise RuntimeError('Invalid PPG sampling rate.')

    window_samples = int(window_seconds * fs)
    step_samples = max(int(step_seconds * fs), 1)
    if window_samples <= 0 or samples.size < window_samples:
        raise RuntimeError('Not enough PPG data to compute an HRV baseline.')

    metrics_windows: List[Dict[str, float]] = []
    for start in range(0, samples.size - window_samples + 1, step_samples):
        segment = samples[start:start + window_samples]
        try:
            metrics_windows.append(compute_hrv_metrics(segment, fs))
        except RuntimeError:
            continue

    if not metrics_windows:
        raise RuntimeError('Unable to compute HRV metrics from the collected samples.')

    return metrics_windows


def summarize_hrv_windows(hrv_windows: Sequence[Dict[str, float]]) -> Dict[str, MetricStats]:
    summary: Dict[str, MetricStats] = {}
    for metric in HRV_METRIC_NAMES:
        values = [window.get(metric) for window in hrv_windows if window.get(metric) is not None]
        summary[metric] = _metric_stats(values)
    return summary


async def collect_sensor_data(duration_seconds: float) -> Tuple[List[Dict[str, Any]], np.ndarray, int]:
    loop = asyncio.get_running_loop()
    eeg_task = loop.run_in_executor(None, lambda: _collect_eeg_payloads(duration_seconds))
    ppg_task = loop.run_in_executor(None, lambda: _collect_ppg_samples(duration_seconds))
    payloads, (ppg_samples, ppg_fs) = await asyncio.gather(eeg_task, ppg_task)
    return payloads, ppg_samples, ppg_fs


class BaselineCalculator:
    def __init__(
        self,
        duration_seconds: float,
        hrv_window_seconds: float,
        hrv_step_seconds: float,
    ) -> None:
        self.duration_seconds = duration_seconds
        self.hrv_window_seconds = hrv_window_seconds
        self.hrv_step_seconds = hrv_step_seconds

    async def calculate(self) -> BaselineResponse:
        payloads, ppg_samples, ppg_fs = await collect_sensor_data(self.duration_seconds)
        metric_series = [payload.get('metrics') for payload in payloads if payload.get('metrics')]
        if not metric_series:
            raise RuntimeError('Unable to compute EEG metrics for baseline calculation.')

        hrv_windows = compute_hrv_windows(
            ppg_samples, ppg_fs, self.hrv_window_seconds, self.hrv_step_seconds
        )

        summary = summarize_metric_series(metric_series)
        summary.update(summarize_hrv_windows(hrv_windows))

        return BaselineResponse(
            duration_seconds=self.duration_seconds,
            sample_count=len(metric_series),
            hrv_window_seconds=self.hrv_window_seconds,
            hrv_window_count=len(hrv_windows),
            metrics=summary,
        )


async def sample_current_metrics(duration_seconds: float) -> Tuple[Dict[str, float], int, float]:
    payloads, ppg_samples, ppg_fs = await collect_sensor_data(duration_seconds)
    metric_series = [payload.get('metrics') for payload in payloads if payload.get('metrics')]
    if not metric_series:
        raise RuntimeError('Unable to compute EEG metrics for monitoring.')

    averages = average_metric_series(metric_series)
    hrv_metrics = compute_hrv_metrics(ppg_samples, ppg_fs)
    averages.update(hrv_metrics)

    hrv_seconds = float(ppg_samples.size / ppg_fs) if ppg_fs else 0.0
    return averages, len(metric_series), hrv_seconds


def compute_deviations(
    current_metrics: Dict[str, float],
    baseline: BaselineResponse,
    threshold: float,
) -> Tuple[Dict[str, float], List[str], float]:
    deviations: Dict[str, float] = {}
    alerts: List[str] = []
    composite_sum = 0.0
    weight_sum = 0.0
    for metric, stats in baseline.metrics.items():
        value = current_metrics.get(metric)
        if value is None:
            continue
        std = stats.std
        if std is not None and std > 1e-6:
            z_score = (value - stats.mean) / std
        else:
            z_score = 0.0
        deviations[metric] = float(z_score)
        if std is None or std <= 1e-6 or math.isnan(z_score):
            continue

        if metric in NEGATIVE_ALERT_METRICS:
            stress_score = z_score  # drops (negative z) should stay negative
        else:
            stress_score = -z_score  # spikes become negative values

        weight = COMPOSITE_WEIGHTS.get(metric)
        if weight:
            composite_sum += stress_score * weight
            weight_sum += weight

        if metric in NEGATIVE_ALERT_METRICS and z_score <= -threshold:
            alerts.append(
                f"{metric} dropped -{abs(z_score):.1f}σ (current={value:.2f}, baseline={stats.mean:.2f})"
            )
        elif metric in POSITIVE_ALERT_METRICS and z_score >= threshold:
            alerts.append(
                f"{metric} spiked +{z_score:.1f}σ (current={value:.2f}, baseline={stats.mean:.2f})"
            )
    composite_score = float(composite_sum / weight_sum) if weight_sum > 0 else 0.0
    return deviations, alerts, composite_score


async def _dispatch_notification(message: str, title: str, delay_seconds: float) -> None:
    await asyncio.sleep(max(0.0, delay_seconds))
    script = f'display notification {json.dumps(message)} with title {json.dumps(title)}'
    try:
        subprocess.run(['osascript', '-e', script], check=True)
    except Exception as exc:  # pragma: no cover
        print(f'Failed to deliver notification: {exc}')


app = FastAPI(title='Muse Metrics API', version='0.1.0')

app.add_middleware(
    CORSMiddleware,
    allow_origins=['*'],
    allow_credentials=False,
    allow_methods=['*'],
    allow_headers=['*'],
)


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


@app.post('/baseline', response_model=BaselineResponse)
async def calculate_baseline_endpoint(request: BaselineRequest) -> BaselineResponse:
    calculator = BaselineCalculator(
        duration_seconds=request.duration_seconds,
        hrv_window_seconds=request.hrv_window_seconds,
        hrv_step_seconds=request.hrv_step_seconds,
    )
    try:
        return await calculator.calculate()
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc


@app.post('/monitor', response_model=MonitorResponse)
async def monitor_with_baseline(request: MonitorRequest) -> MonitorResponse:
    try:
        current_metrics, sample_count, hrv_seconds = await sample_current_metrics(
            request.duration_seconds
        )
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    deviations, alerts, composite_score = compute_deviations(
        current_metrics=current_metrics,
        baseline=request.baseline,
        threshold=request.deviation_threshold,
    )

    return MonitorResponse(
        timestamp=time.time(),
        metrics=current_metrics,
        deviations=deviations,
        alerts=alerts,
        sample_count=sample_count,
        hrv_sample_seconds=hrv_seconds,
        composite_score=composite_score,
    )


@app.post('/notify')
async def schedule_notification(request: NotificationRequest, background_tasks: BackgroundTasks) -> Dict[str, Any]:
    if request.message.strip() == '':
        raise HTTPException(status_code=400, detail='Message must not be empty.')
    background_tasks.add_task(
        _dispatch_notification,
        request.message,
        request.title,
        request.delay_seconds,
    )
    return {'status': 'scheduled', 'delay_seconds': request.delay_seconds}


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
