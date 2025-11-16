import { useEffect, useMemo, useRef, useState } from 'react';
import LiveWaves from './components/LiveWaves.jsx';
import HRVDeviationChart from './components/HRVDeviationChart.jsx';
import CompassIcon from './components/CompassIcon.jsx';
import { useBaselineMonitor } from './hooks/useBaselineMonitor.js';
import { useMuseStream } from './hooks/useMuseStream.js';
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
const NEGATIVE_ALERT_METRICS = new Set(['hrv_rmssd', 'hrv_sdnn']);
const POSITIVE_ALERT_METRICS = new Set([
  'alpha_relaxation',
  'theta_relaxation',
  'beta_concentration',
  'hrv_lf_hf',
]);

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

const METRIC_HINTS = {
  alpha_relaxation: {
    label: 'Alpha calm',
    hint: 'Higher = calm, eyes-open relaxation',
  },
  theta_relaxation: {
    label: 'Theta relaxation',
    hint: 'Higher = deeper relaxation/creativity',
  },
  beta_concentration: {
    label: 'Beta concentration',
    hint: 'Higher = analytical engagement',
  },
  hrv_rmssd: {
    label: 'RMSSD',
    hint: 'Higher = resilient parasympathetic tone',
  },
  hrv_sdnn: {
    label: 'SDNN',
    hint: 'Higher = overall HRV stability',
  },
  hrv_lf_hf: {
    label: 'LF/HF',
    hint: 'Higher = sympathetic/stress load',
  },
};

function metricLabel(key) {
  return METRIC_HINTS[key]?.label || readableMetricName(key);
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
  const [clientId, setClientId] = useState(null);
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
    clientId,
  });
  const liveStream = useMuseStream(WS_URL);
  const [breathingPhase, setBreathingPhase] = useState(BREATHING_PHASES[0]);
  const breathingTimeoutRef = useRef(null);
  const deviationNotificationRef = useRef(null);
  const [notificationPermission, setNotificationPermission] = useState(() => {
    if (typeof window === 'undefined' || !('Notification' in window)) {
      return 'unsupported';
    }
    return Notification.permission;
  });
  const [calendarStatus, setCalendarStatus] = useState('disconnected');
  const [calendarEvents, setCalendarEvents] = useState([]);
  const [agentMessage, setAgentMessage] = useState(null);
  const [threshold, setThreshold] = useState(1.5);

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
    if (typeof window === 'undefined') {
      return;
    }
    const key = 'stressCompassClientId';
    let id = window.localStorage.getItem(key);
    if (!id) {
      if (window.crypto?.randomUUID) {
        id = window.crypto.randomUUID();
      } else {
        id = `client-${Date.now()}`;
      }
      window.localStorage.setItem(key, id);
    }
    setClientId(id);
  }, []);

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
      return;
    }
    Notification.requestPermission()
      .then((permission) => {
        setNotificationPermission(permission);
      })
      .catch(() => {
        setNotificationPermission(Notification.permission);
      });
  };

  const fetchThresholdValue = async () => {
    if (!clientId) {
      return;
    }
    try {
      const response = await fetch(`${API_URL}/agent/threshold?client_id=${clientId}`);
      if (response.ok) {
        const data = await response.json();
        if (typeof data.value === 'number') {
          setThreshold(data.value);
        }
      }
    } catch (error) {
      console.error('Failed to fetch threshold', error);
    }
  };

  const updateThresholdValue = async (value) => {
    if (!clientId) {
      return;
    }
    try {
      setThreshold(value);
      await fetch(`${API_URL}/agent/threshold?client_id=${clientId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ value }),
      });
    } catch (error) {
      console.error('Failed to update threshold', error);
    }
  };

  const fetchCalendarEvents = async () => {
    if (!clientId) {
      return;
    }
    try {
      setCalendarStatus('connecting');
      const response = await fetch(`${API_URL}/calendar/events?client_id=${clientId}`);
      if (!response.ok) {
        throw new Error((await response.json()).detail || 'Failed to read calendar events');
      }
      const data = await response.json();
      setCalendarEvents(data.events || []);
      setCalendarStatus('connected');
      fetchThresholdValue();
    } catch (error) {
      console.error('Calendar fetch failed', error);
      setCalendarStatus('error');
    }
  };

  const triggerAgent = async () => {
    if (!monitorResult?.composite_score || !clientId) {
      return;
    }
    try {
      setAgentMessage('Running agent...');
      const response = await fetch(`${API_URL}/agent/mock-trigger?client_id=${clientId}`, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ composite_score: monitorResult.composite_score }),
      });
      if (!response.ok) {
        throw new Error((await response.json()).detail || 'Agent trigger failed');
      }
      const data = await response.json();
      const statusText =
        data.status === 'scheduled'
          ? `Intervention scheduled in ${Math.round(data.delay_seconds)}s.`
          : 'Intervention sent.';
      setAgentMessage(`${statusText} Message: ${data.message}`);
      if (data.current_event) {
        setCalendarEvents((prev) => {
          if (prev?.length) {
            return prev;
          }
          return [data.current_event, data.next_event].filter(Boolean);
        });
      }
    } catch (error) {
      console.error('Agent trigger failed', error);
      setAgentMessage('Failed to trigger agent.');
    }
  };

  const connectCalendar = async () => {
    if (!clientId) {
      return;
    }
    try {
      setCalendarStatus('connecting');
      const response = await fetch(`${API_URL}/calendar/connect/start?client_id=${clientId}`);
      if (!response.ok) {
        throw new Error('Failed to start OAuth flow');
      }
      const data = await response.json();
      const authWindow = window.open(data.auth_url, '_blank', 'width=500,height=700');
      const poll = window.setInterval(async () => {
        try {
          const statusRes = await fetch(`${API_URL}/calendar/status?client_id=${clientId}`);
          const statusData = await statusRes.json();
          if (statusData.connected) {
            window.clearInterval(poll);
            if (authWindow) {
              authWindow.close();
            }
            setCalendarStatus('connected');
            fetchCalendarEvents();
            fetchThresholdValue();
          }
        } catch (error) {
          console.error('Status check failed', error);
        }
      }, 2000);
    } catch (error) {
      console.error('Calendar connect failed', error);
      setCalendarStatus('error');
    }
  };

  useEffect(() => {
    if (!clientId) {
      return;
    }
    fetch(`${API_URL}/calendar/status?client_id=${clientId}`)
      .then((res) => res.json())
      .then((data) => {
        if (data.connected) {
          setCalendarStatus('connected');
          fetchCalendarEvents();
          fetchThresholdValue();
        }
      })
      .catch(() => {});
  }, [clientId]);

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

  return (
    <div className="app-shell">
      <header className="app-header">
        <div className="logo-mark">
          <CompassIcon size={52} />
        </div>
        <div className="app-header-text">
          <h1>Stress Compass</h1>
          <p>regular awareness for calmer days.</p>
        </div>
      </header>

      <div className="card baseline-card">
        <h2>Baseline Calibration</h2>
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
        )}
        {isCalculatingBaseline && (
          <div className="baseline-visualizer active">
            <div className="baseline-visualizer-inner">
              <LiveWaves
                url={WS_URL}
                width={720}
                height={240}
                showStatus={false}
                showNotifications={false}
                showLegend={false}
                showMetricGrid={false}
                className="compact-waves"
                enabled
              />
              <div
                className="baseline-breath-shadow active"
                data-phase={breathingPhase.key}
              />
              <div className="baseline-visualizer-overlay active">
                <div className="baseline-timer">
                  <div
                    className="baseline-timer-ring"
                    style={{ background: `conic-gradient(#22d3ee ${baselineProgressPercent * 3.6}deg, rgba(148, 163, 184, 0.2) 0deg)` }}
                  >
                    <span>{formatDuration(Math.max(BASELINE_DURATION - baselineElapsed, 0))}</span>
                  </div>
                  <div className="baseline-timer-label">Time left</div>
                </div>
                <div>
                  <div className="overlay-title">Recording baseline...</div>
                  <div className="overlay-subtitle">Stay relaxed and keep still</div>
                </div>
              </div>
            </div>
            <div className="baseline-visualizer-caption">
              <span>Slow, steady breaths</span>
              <span>{formatDuration(baselineElapsed)} / {formatDuration(BASELINE_DURATION)}</span>
            </div>
          </div>
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
                  <div className="metric-label">{metricLabel(key)}</div>
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
            <button
              type="button"
              className="ghost"
              onClick={async () => {
                try {
                  await fetch(`${API_URL}/notify`, {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({
                      message: 'Stress Compass test notification',
                      delay_seconds: 0,
                    }),
                  });
                } catch (error) {
                  console.error('Failed to request test notification', error);
                }
              }}
            >
              Test desktop notification
            </button>
          )}
          {notificationPermission === 'denied' && (
            <span className="muted small">Allow notifications in your browser settings.</span>
          )}
        </div>
        <div className="calendar-settings">
          <span>Google Calendar:</span>
          <span className={`badge ${calendarStatus === 'connected' ? 'badge-success' : ''} ${calendarStatus === 'error' ? 'badge-error' : ''}`.trim()}>
            {calendarStatus === 'connected'
              ? 'Connected'
              : calendarStatus === 'connecting'
                ? 'Connecting...'
                : calendarStatus === 'error'
                  ? 'Failed'
                  : 'Disconnected'}
          </span>
          <button
            type="button"
            className="ghost"
            onClick={calendarStatus === 'connected' ? fetchCalendarEvents : connectCalendar}
            disabled={!clientId}
          >
            {calendarStatus === 'connected' ? 'Refresh calendar' : 'Connect calendar'}
          </button>
        </div>
        {agentMessage && <p className="muted" style={{ marginBottom: '0.5rem' }}>{agentMessage}</p>}
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
                  <div className="metric-label">{metricLabel(key)}</div>
                  <div className="metric-value">{formatNumber(value)}</div>
                  <div className="metric-subvalue">{formatDeviation(deviation)}</div>
                  {METRIC_HINTS[key]?.hint && (
                    <div className="metric-hint">{METRIC_HINTS[key].hint}</div>
                  )}
                </div>
              );
            })}
          </div>
        ) : (
          <p className="muted">Baseline calibration must finish before deviation detection starts.</p>
        )}
        {calendarEvents.length > 0 && (
          <div className="calendar-events">
            <h3>Today's events</h3>
            <ul>
              {calendarEvents.slice(0, 3).map((event) => (
                <li key={`${event.summary}-${event.start}`}>
                  <strong>{event.summary}</strong>
                  <span>
                    {new Date(event.start).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                    {' - '}
                    {new Date(event.end).toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
                  </span>
                </li>
              ))}
            </ul>
          </div>
        )}
      </div>

      <div className="card insights-card">
        <h2>Mind-State Deviation Insights</h2>
        <p className="muted">
          {monitorHistory.length
            ? `Tracking combined neuro + HRV deviation over the last ${hrvWindowLabel}.`
            : 'Begin monitoring to populate deviation trends.'}
        </p>
        <HRVDeviationChart
          history={monitorHistory}
          highlightValue={monitorResult?.composite_score ?? null}
        />
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
          <LiveWaves url={WS_URL} stream={liveStream} />
        </div>
      </div>

      <div className="card debug-card">
        <h3>Agent Debug</h3>
        <p className="muted">
          Use this only if you need to simulate a stress event manually. Real triggers fire automatically when the stress index is high.
        </p>
        <div className="manual-score">
          <label htmlFor="threshold-input">Trigger threshold (σ):</label>
          <input
            id="threshold-input"
            type="number"
            min="0"
            max="5"
            step="0.1"
            value={threshold}
            onChange={(event) => setThreshold(Number(event.target.value) || 0)}
            onBlur={(event) => updateThresholdValue(Number(event.target.value) || 0)}
          />
          <span className="muted small">Auto trigger fires when stress index ≥ this value.</span>
        </div>
        <button
          type="button"
          className="ghost"
          onClick={triggerAgent}
          disabled={!monitorResult?.composite_score || calendarStatus !== 'connected'}
        >
          Mock stress trigger
        </button>
      </div>
    </div>
  );
}
