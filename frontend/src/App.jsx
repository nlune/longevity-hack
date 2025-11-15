import { useEffect, useMemo, useRef, useState } from 'react';
import LiveWaves from './components/LiveWaves.jsx';
import HRVDeviationChart from './components/HRVDeviationChart.jsx';
import { useBaselineMonitor } from './hooks/useBaselineMonitor.js';
import { deriveMindStates } from './utils/mindState.js';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/metrics';
const API_URL = import.meta.env.VITE_API_URL || 'http://localhost:8000';
const BASELINE_DURATION = 60;
const MONITOR_DURATION = 30;
const MONITOR_INTERVAL = 5000;
const BREATHING_PHASES = [
  { key: 'inhale', duration: 4000 },
  { key: 'hold', duration: 2000 },
  { key: 'exhale', duration: 4000 },
  { key: 'hold', duration: 2000 },
];
const HEARTBEAT_INTERVAL_MS = 60000;
const NEGATIVE_ALERT_METRICS = new Set([
  'alpha_relaxation',
  'theta_relaxation',
  'beta_concentration',
  'hrv_rmssd',
  'hrv_sdnn',
]);
const POSITIVE_ALERT_METRICS = new Set(['hrv_lf_hf']);

function formatNumber(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '--';
  }
  return Number(value).toFixed(2);
}

function formatDeviation(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '0.0σ';
  }
  return `${value >= 0 ? '+' : ''}${value.toFixed(1)}σ`;
}

function getDeviationBadge(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '';
  }
  if (value >= 2) {
    return 'high-positive';
  }
  if (value <= -2) {
    return 'high-negative';
  }
  if (Math.abs(value) >= 1) {
    return 'medium';
  }
  return 'low';
}

function readableMetricName(name) {
  return name.replace(/_/g, ' ');
}

function metricDirection(metric) {
  if (NEGATIVE_ALERT_METRICS.has(metric)) {
    return -1; // higher is better, lower is concerning
  }
  if (POSITIVE_ALERT_METRICS.has(metric)) {
    return 1; // higher is concerning
  }
  return 0; // neutral -> treat magnitude only
}

function isBeneficialDeviation(metric, deviation) {
  if (deviation === undefined || deviation === null || Number.isNaN(deviation)) {
    return false;
  }
  const direction = metricDirection(metric);
  if (direction === -1) {
    return deviation > 0.2;
  }
  if (direction === 1) {
    return deviation < -0.2;
  }
  return deviation < -0.5;
}

export default function App() {
  const {
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
  } = useBaselineMonitor(API_URL, {
    baselineDuration: BASELINE_DURATION,
    monitorDuration: MONITOR_DURATION,
    monitorInterval: MONITOR_INTERVAL,
    historyLimit: 240,
  });
  const [breathingPhase, setBreathingPhase] = useState(BREATHING_PHASES[0]);
  const breathingTimeoutRef = useRef(null);
  const deviationNotificationRef = useRef(null);
  const [notificationPermission, setNotificationPermission] = useState(() => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      return 'unsupported';
    }
    return Notification.permission;
  });
  const lastTestNotificationRef = useRef(0);
  const heartbeatIntervalRef = useRef(null);
  const [heartbeatEnabled, setHeartbeatEnabled] = useState(false);
  const [notificationInfo, setNotificationInfo] = useState('');

  const monitorReady = Boolean(baseline && monitorResult);
  const monitorStatusLabel = {
    idle: 'Waiting for baseline calibration',
    running: 'Collecting a fresh live sample (~30s)',
    ready: 'Up to date with the most recent live window',
    error: 'Monitoring failed – will retry',
  }[monitorStatus] || 'Idle';

  const baselineProgressPercent = Math.min(Math.round(baselineProgress * 100), 100);
  const mindStates = useMemo(() => deriveMindStates(monitorResult?.deviations), [monitorResult]);
  const hrvWindowLabel = useMemo(() => {
    if (!monitorHistory.length) {
      return '0 min';
    }
    const first = monitorHistory[0]?.timestamp;
    const last = monitorHistory[monitorHistory.length - 1]?.timestamp;
    if (!first || !last || last <= first) {
      const approxMinutes = (monitorHistory.length * MONITOR_INTERVAL) / 60000;
      return `${Math.max(1, Math.round(approxMinutes))} min`;
    }
    const minutes = (last - first) / 60;
    if (minutes >= 1) {
      return `${Math.max(1, Math.round(minutes))} min`;
    }
    const seconds = Math.max(10, Math.round(minutes * 60));
    return `${seconds} sec`;
  }, [monitorHistory]);

  function formatDuration(seconds) {
    const minutes = Math.floor(seconds / 60)
      .toString()
      .padStart(2, '0');
    const remainder = Math.floor(seconds % 60)
      .toString()
      .padStart(2, '0');
    return `${minutes}:${remainder}`;
  }

  useEffect(() => {
    const clearBreathingTimeout = () => {
      if (breathingTimeoutRef.current) {
        window.clearTimeout(breathingTimeoutRef.current);
        breathingTimeoutRef.current = null;
      }
    };

    if (!isCalculatingBaseline) {
      clearBreathingTimeout();
      setBreathingPhase(BREATHING_PHASES[0]);
      return () => clearBreathingTimeout();
    }

    let phaseIndex = 0;
    const advancePhase = () => {
      const phase = BREATHING_PHASES[phaseIndex];
      setBreathingPhase(phase);
      breathingTimeoutRef.current = window.setTimeout(() => {
        phaseIndex = (phaseIndex + 1) % BREATHING_PHASES.length;
        advancePhase();
      }, phase.duration);
    };
    advancePhase();

    return () => clearBreathingTimeout();
  }, [isCalculatingBaseline]);

  useEffect(() => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      setNotificationPermission('unsupported');
      return;
    }
    setNotificationPermission(Notification.permission);
  }, []);

  const requestNotificationPermission = () => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      setNotificationPermission('unsupported');
      setNotificationInfo('Notifications are not supported in this browser.');
      return;
    }
    Notification.requestPermission()
      .then((permission) => {
        setNotificationPermission(permission);
        setNotificationInfo(
          permission === 'granted'
            ? 'Desktop alerts enabled. You can now use test or minute ping.'
            : permission === 'denied'
              ? 'Notifications blocked. Allow them in the browser settings.'
              : 'Permission request dismissed.'
        );
      })
      .catch(() => {
        const fallback = Notification.permission;
        setNotificationPermission(fallback);
        setNotificationInfo('Unable to request notification permission.');
      });
  };

  const ensurePermission = () => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      setNotificationInfo('Notifications are not supported in this browser.');
      return false;
    }
    if (notificationPermission !== 'granted') {
      setNotificationInfo('Enable alerts first, then try again.');
      return false;
    }
    return true;
  };

  const triggerTestNotification = () => {
    if (!ensurePermission()) {
      return;
    }
    const now = Date.now();
    if (now - lastTestNotificationRef.current < 2000) {
      return;
    }
    lastTestNotificationRef.current = now;
    try {
      new Notification('Notification test', {
        body: 'If you can read this, desktop alerts are ready.',
        tag: 'deviation-test',
      });
    } catch (error) {
      console.error('Unable to show test notification', error);
    }
  };

  const toggleHeartbeat = () => {
    if (heartbeatEnabled) {
      setHeartbeatEnabled(false);
      return;
    }
    if (!ensurePermission()) {
      return;
    }
    setHeartbeatEnabled(true);
  };

  useEffect(() => {
    if (!monitorResult?.alerts?.length) {
      return;
    }
    if (typeof window === 'undefined' || !('Notification' in window)) {
      return;
    }
    if (notificationPermission !== 'granted') {
      return;
    }
    const signature = `${monitorResult.timestamp}:${monitorResult.alerts.join('|')}`;
    if (deviationNotificationRef.current === signature) {
      return;
    }
    deviationNotificationRef.current = signature;
    monitorResult.alerts.forEach((alert) => {
      try {
        new Notification('Deviation detected', {
          body: alert,
          tag: `monitor-${signature}`,
          silent: false,
        });
      } catch (error) {
        console.error('Failed to show deviation notification', error);
      }
    });
  }, [monitorResult, notificationPermission]);

  useEffect(() => {
    if (notificationPermission !== 'granted' && heartbeatEnabled) {
      setHeartbeatEnabled(false);
    }
  }, [notificationPermission, heartbeatEnabled]);

  useEffect(() => {
    if (!heartbeatEnabled || notificationPermission !== 'granted') {
      if (heartbeatIntervalRef.current) {
        window.clearInterval(heartbeatIntervalRef.current);
        heartbeatIntervalRef.current = null;
      }
      return;
    }
    heartbeatIntervalRef.current = window.setInterval(() => {
      try {
        new Notification('Monitoring pulse', {
          body: `Still watching metrics (${new Date().toLocaleTimeString()})`,
          tag: 'monitor-heartbeat',
          silent: true,
        });
      } catch (error) {
        console.error('Failed to send heartbeat notification', error);
      }
    }, HEARTBEAT_INTERVAL_MS);
    return () => {
      if (heartbeatIntervalRef.current) {
        window.clearInterval(heartbeatIntervalRef.current);
        heartbeatIntervalRef.current = null;
      }
    };
  }, [heartbeatEnabled, notificationPermission]);

  return (
    <div className="app-shell">
      <div className="card baseline-card">
        <h1>Baseline Calibration</h1>
        <p>
          We run a one-minute capture to understand your personal alpha/beta ratios and heart-rate variability. Keep the
          headset steady and relax while the baseline is recorded.
        </p>

        <div className="baseline-actions">
          <button type="button" onClick={calculateBaseline} disabled={isCalculatingBaseline}>
            {baseline ? 'Recalculate baseline' : 'Start calibration'}
          </button>
          {isCalculatingBaseline && (
            <span className="muted">
              Collecting baseline ({baselineProgressPercent}% of {formatDuration(BASELINE_DURATION)})
            </span>
          )}
        </div>

        {isCalculatingBaseline && (
          <>
            <div className="baseline-progress">
              <div className="progress-bar">
                <div
                  className="progress-bar-fill"
                  style={{ width: `${baselineProgressPercent}%` }}
                />
              </div>
              <div className="progress-label">
                <span>{formatDuration(baselineElapsed)}</span>
                <span>{formatDuration(BASELINE_DURATION)}</span>
              </div>
            </div>
            <div className="baseline-visualizer">
              <div className="baseline-visualizer-inner">
                {isCalculatingBaseline && (
                  <LiveWaves
                    url={WS_URL}
                    width={720}
                    height={240}
                    showStatus={false}
                    showNotifications={false}
                    showLegend={false}
                    showMetricGrid={false}
                    className="compact-waves"
                  />
                )}
                <div className="baseline-breath-shadow" data-phase={breathingPhase.key} />
              </div>
              <div className="baseline-visualizer-caption">
                <span>Slow, steady breaths</span>
                <span>{formatDuration(baselineElapsed)} / {formatDuration(BASELINE_DURATION)}</span>
              </div>
            </div>
          </>
        )}

        {baselineError && <p className="error">{baselineError}</p>}

        {baseline && (
          <>
            <p className="muted">
              Last calibration used {baseline.sample_count} EEG epochs and {baseline.hrv_window_count} HRV windows.
            </p>
            <div className="baseline-grid">
              {Object.entries(baseline.metrics).map(([key, stats]) => (
                <div key={key} className="metric compact">
                  <div className="metric-label">{readableMetricName(key)}</div>
                  <div className="metric-value">{formatNumber(stats.mean)}</div>
                  <div className="metric-subvalue">σ {formatNumber(stats.std)}</div>
                </div>
              ))}
            </div>
          </>
        )}
      </div>

      <div className="card monitor-card">
        <h2>Live Deviation Monitor</h2>
        <p className="muted">
          {monitorStatusLabel}
          {monitorResult?.timestamp && ` • Updated ${new Date(monitorResult.timestamp * 1000).toLocaleTimeString()}`}
        </p>
        <div className="notification-settings">
          <span>Desktop alerts:</span>
          <span
            className={`badge ${notificationPermission === 'granted' ? 'badge-success' : ''} ${notificationPermission === 'denied' ? 'badge-error' : ''}`.trim()}
          >
            {notificationPermission === 'unsupported'
              ? 'Unavailable'
              : notificationPermission === 'granted'
                ? 'Enabled'
                : notificationPermission === 'denied'
                  ? 'Blocked'
                  : 'Tap to enable'}
          </span>
          {notificationPermission === 'default' && (
            <button type="button" className="ghost" onClick={requestNotificationPermission}>
              Enable alerts
            </button>
          )}
          {notificationPermission === 'granted' && (
            <button type="button" className="ghost" onClick={triggerTestNotification}>
              Test notification
            </button>
          )}
          {notificationPermission === 'granted' && (
            <button
              type="button"
              className={`ghost ${heartbeatEnabled ? 'active' : ''}`.trim()}
              onClick={toggleHeartbeat}
            >
              Minute ping: {heartbeatEnabled ? 'On' : 'Off'}
            </button>
          )}
          {notificationPermission === 'denied' && (
            <span className="muted small">Allow notifications in your browser settings.</span>
          )}
          {notificationInfo && (
            <span className="muted small" style={{ flexBasis: '100%' }}>
              {notificationInfo}
            </span>
          )}
        </div>
        {monitorError && <p className="error">{monitorError}</p>}
        {monitorResult?.alerts?.length > 0 && (
          <div className="alert-panel">
            {monitorResult.alerts.map((alert) => (
              <div key={alert} className="alert-item alert-negative">
                {alert}
              </div>
            ))}
          </div>
        )}
        {monitorReady ? (
          <div className="monitor-grid">
            {Object.entries(monitorResult.metrics).map(([key, value]) => {
              const deviation = monitorResult.deviations?.[key];
              const badge = getDeviationBadge(deviation);
              const positiveEvent = isBeneficialDeviation(key, deviation);
              return (
                <div
                  key={key}
                  className={`metric deviation ${badge} ${positiveEvent ? 'deviation-positive' : ''}`.trim()}
                >
                  <div className="metric-label">{readableMetricName(key)}</div>
                  <div className="metric-value">{formatNumber(value)}</div>
                  <div className="metric-subvalue">{formatDeviation(deviation)}</div>
                </div>
              );
            })}
          </div>
        ) : (
          <p className="muted">Baseline calibration must finish before deviation detection starts.</p>
        )}
      </div>

      <div className="card insights-card">
        <h2>Mind-State Deviation Insights</h2>
        <p className="muted">
          {monitorHistory.length
            ? `Tracking combined neuro + HRV deviation over the last ${hrvWindowLabel}.`
            : 'Begin monitoring to populate deviation trends.'}
        </p>
        <HRVDeviationChart history={monitorHistory} />
        <div className="mindstate-grid">
          {mindStates.map((state) => (
            <div key={state.key} className={`mindstate-card ${state.key}`}>
              <div className="mindstate-header">
                <span>{state.label}</span>
                <span className="mindstate-status">{state.status}</span>
              </div>
              <div className="mindstate-score">
                {state.percent}
                <span>%</span>
              </div>
              <div className="mindstate-meter">
                <div className="mindstate-meter-fill" style={{ width: `${state.percent}%` }} />
              </div>
              <p className="mindstate-description">{state.description}</p>
            </div>
          ))}
        </div>
      </div>

      <div className="card graph-card">
        <h2>Muse Metrics Stream</h2>
        <p>Live band powers and derived relaxation/concentration metrics from the FastAPI backend.</p>
        <div className="graph-visualizer">
          {isCalculatingBaseline ? (
            <div className="graph-placeholder">
              <span className="pulse-dot" />
              <div>
                <div className="overlay-title">Baseline capture in progress</div>
                <div className="overlay-subtitle">We will resume the main stream once calibration ends.</div>
              </div>
            </div>
          ) : (
            <LiveWaves url={WS_URL} />
          )}
        </div>
      </div>
    </div>
  );
}
