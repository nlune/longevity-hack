import { useCallback, useEffect, useRef, useState } from 'react';

const DEFAULT_MONITOR_DURATION = 30;
const DEFAULT_MONITOR_INTERVAL = 5000;
const DEFAULT_DEVIATION_THRESHOLD = 2;
const DEFAULT_BASELINE_DURATION = 120;
const DEFAULT_HRV_WINDOW_SECONDS = 30;
const DEFAULT_HRV_STEP_SECONDS = 10;

function parseError(response) {
  return response?.detail || response?.message || response?.error || 'Unexpected server response';
}

export function useBaselineMonitor(
  apiUrl,
  {
    autoStart = true,
    baselineDuration = DEFAULT_BASELINE_DURATION,
    monitorDuration = DEFAULT_MONITOR_DURATION,
    monitorInterval = DEFAULT_MONITOR_INTERVAL,
    deviationThreshold = DEFAULT_DEVIATION_THRESHOLD,
    hrvWindowSeconds = DEFAULT_HRV_WINDOW_SECONDS,
    hrvStepSeconds = DEFAULT_HRV_STEP_SECONDS,
    historyLimit = 180,
    clientId,
  } = {}
) {
  const [baseline, setBaseline] = useState(null);
  const [isCalculatingBaseline, setIsCalculatingBaseline] = useState(false);
  const [baselineError, setBaselineError] = useState(null);
  const [monitorResult, setMonitorResult] = useState(null);
  const [monitorHistory, setMonitorHistory] = useState([]);
  const [monitorError, setMonitorError] = useState(null);
  const [monitorStatus, setMonitorStatus] = useState('idle');
  const baselineAbortRef = useRef(null);
  const baselineTimerRef = useRef(null);
  const [baselineProgress, setBaselineProgress] = useState(0);
  const [baselineElapsed, setBaselineElapsed] = useState(0);

  const stopBaselineTimer = useCallback(() => {
    if (baselineTimerRef.current) {
      window.clearInterval(baselineTimerRef.current);
      baselineTimerRef.current = null;
    }
  }, []);

  const startBaselineTimer = useCallback(() => {
    stopBaselineTimer();
    const startedAt = Date.now();
    setBaselineProgress(0);
    setBaselineElapsed(0);
    baselineTimerRef.current = window.setInterval(() => {
      const elapsedSeconds = (Date.now() - startedAt) / 1000;
      setBaselineElapsed(Math.min(elapsedSeconds, baselineDuration));
      setBaselineProgress(Math.min(elapsedSeconds / baselineDuration, 0.995));
    }, 200);
  }, [baselineDuration, stopBaselineTimer]);

  const calculateBaseline = useCallback(async () => {
    setBaseline(null);
    setMonitorResult(null);
    setMonitorHistory([]);
    setBaselineError(null);
    setMonitorError(null);
    setMonitorStatus('idle');
    setIsCalculatingBaseline(true);

    baselineAbortRef.current?.abort();
    const controller = new AbortController();
    baselineAbortRef.current = controller;
    startBaselineTimer();

    try {
      const response = await fetch(`${apiUrl}/baseline`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
          duration_seconds: baselineDuration,
          hrv_window_seconds: hrvWindowSeconds,
          hrv_step_seconds: hrvStepSeconds,
        }),
        signal: controller.signal,
      });

      if (!response.ok) {
        const errorPayload = await response.json().catch(() => ({}));
        throw new Error(errorPayload.detail || 'Failed to calculate baseline');
      }

      const data = await response.json();
      setBaseline(data);
      return data;
    } catch (error) {
      if (error.name === 'AbortError') {
        return null;
      }
      setBaselineError(error.message || 'Baseline calculation failed');
      throw error;
    } finally {
      if (baselineAbortRef.current === controller) {
        baselineAbortRef.current = null;
      }
      stopBaselineTimer();
      setIsCalculatingBaseline(false);
      setBaselineProgress(0);
      setBaselineElapsed(0);
    }
  }, [apiUrl, baselineDuration, hrvStepSeconds, hrvWindowSeconds, startBaselineTimer, stopBaselineTimer]);

  useEffect(() => {
    if (!autoStart || baseline || isCalculatingBaseline) {
      return;
    }
    calculateBaseline().catch(() => {});
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [autoStart]);

  useEffect(() => () => {
    baselineAbortRef.current?.abort();
    stopBaselineTimer();
  }, [stopBaselineTimer]);

  useEffect(() => {
    if (!baseline) {
      return undefined;
    }

    let cancelled = false;
    let timeoutId = null;

    async function runMonitor() {
      setMonitorStatus('running');
      setMonitorError(null);

      try {
        const url = clientId ? `${apiUrl}/monitor?client_id=${clientId}` : `${apiUrl}/monitor`;
        const response = await fetch(url, {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({
            baseline,
            duration_seconds: monitorDuration,
            deviation_threshold: deviationThreshold,
          }),
        });

        if (!response.ok) {
          const payload = await response.json().catch(() => ({}));
          throw new Error(parseError(payload));
        }

        const data = await response.json();
        if (!cancelled) {
          setMonitorResult(data);
          setMonitorHistory((prev) => {
            const next = [...prev, data];
            if (next.length > historyLimit) {
              next.splice(0, next.length - historyLimit);
            }
            return next;
          });
          setMonitorStatus('ready');
        }
      } catch (error) {
        if (cancelled) {
          return;
        }
        setMonitorStatus('error');
        setMonitorError(error.message || 'Monitoring failed');
      }

      if (!cancelled) {
        timeoutId = window.setTimeout(runMonitor, monitorInterval);
      }
    }

    runMonitor();

    return () => {
      cancelled = true;
      if (timeoutId) {
        window.clearTimeout(timeoutId);
      }
    };
  }, [apiUrl, baseline, monitorDuration, monitorInterval, deviationThreshold]);

  return {
    baseline,
    baselineError,
    isCalculatingBaseline,
    calculateBaseline,
    monitorResult,
    monitorHistory,
    monitorError,
    monitorStatus,
    baselineProgress,
    baselineElapsed,
  };
}
