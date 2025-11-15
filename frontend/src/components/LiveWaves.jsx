import { useMemo } from 'react';
import { DEFAULT_BANDS, useMuseStream } from '../hooks/useMuseStream.js';
import { useConcentrationNotifications } from '../hooks/useConcentrationNotifications.js';

const BAND_COLORS = {
  delta: '#38bdf8',
  theta: '#a855f7',
  alpha: '#22d3ee',
  beta: '#f472b6',
};

function formatNumber(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return '--';
  }
  return Number(value).toFixed(2);
}

export default function LiveWaves({
  url,
  width = 800,
  height = 300,
  showStatus = true,
  showNotifications = true,
  showLegend = true,
  showMetricGrid = true,
  className = '',
}) {
  const { bandSeries, latest, status, error } = useMuseStream(url);

  // Enable browser notifications for low concentration
  const { permissionStatus, lastAlertTime } = useConcentrationNotifications(latest, 0);

  const paths = useMemo(() => {
    const resolved = {};
    DEFAULT_BANDS.forEach((band) => {
      const series = bandSeries[band] || [];
      if (!series.length) {
        return;
      }
      const values = series.map((point) => point.value ?? 0);
      const min = Math.min(...values);
      const max = Math.max(...values);
      const span = max - min || 1;
      const step = series.length > 1 ? width / (series.length - 1) : width;
      const path = series
        .map((point, index) => {
          const normalized = (point.value - min) / span;
          const x = index * step;
          const y = height - normalized * height;
          return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
        })
        .join(' ');
      resolved[band] = { path, min, max };
    });
    return resolved;
  }, [bandSeries, width, height]);

  const statusLabel = {
    connecting: 'Connecting to backend...',
    online: 'Streaming from backend',
    disconnected: 'Disconnected',
    error: 'Connection error',
  }[status] || 'Idle';

  return (
    <div className={`live-waves ${className}`.trim()}>
      {showStatus && (
        <div className="status">
          <span className={`status-dot ${status === 'online' ? 'online' : ''}`} />
          <span>{statusLabel}</span>
          {latest?.mode === 'mock' && (
            <span className="badge" title="Streaming simulated data">
              Mock data
            </span>
          )}
        </div>
      )}

      {showNotifications && (
        <div className="status" style={{ marginTop: showStatus ? '0.5rem' : 0 }}>
          <span>🔔 Notifications: </span>
          <span style={{
            color: permissionStatus === 'granted' ? '#22c55e' :
                   permissionStatus === 'denied' ? '#dc2626' : '#fbbf24',
            fontWeight: 'bold'
          }}>
            {permissionStatus === 'granted' ? 'Enabled' :
             permissionStatus === 'denied' ? 'Blocked' :
             permissionStatus === 'requesting' ? 'Requesting...' : 'Checking...'}
          </span>
          {permissionStatus === 'granted' && (
            <button
              onClick={() => {
                new Notification('Test Notification 🧪', {
                  body: 'If you see this, notifications are working!',
                  tag: 'test-notification',
                });
              }}
              style={{
                marginLeft: '0.5rem',
                padding: '0.25rem 0.5rem',
                fontSize: '0.85em',
                cursor: 'pointer',
                borderRadius: '4px',
                border: '1px solid #ccc',
                background: '#f0f0f0',
              }}
            >
              Test Notification
            </button>
          )}
          {lastAlertTime && (
            <span className="badge" style={{ marginLeft: '0.5rem' }}>
              Last alert: {lastAlertTime}
            </span>
          )}
          {latest?.metrics?.beta_concentration !== undefined && (
            <span style={{ marginLeft: '0.5rem', fontSize: '0.9em', opacity: 0.8 }}>
              (Beta: {formatNumber(latest.metrics.beta_concentration)})
            </span>
          )}
        </div>
      )}

      {error && (
        <p style={{ color: '#dc2626', marginTop: '0.5rem' }}>{error}</p>
      )}

      <div className={`chart-wrapper ${!showLegend && !showMetricGrid ? 'chart-wrapper-compact' : ''}`}>
        <svg viewBox={`0 0 ${width} ${height}`} className="chart-svg" role="img">
          <defs>
            <linearGradient id="gridGradient" x1="0" y1="0" x2="0" y2="1">
              <stop offset="0%" stopColor="rgba(148, 163, 184, 0.25)" />
              <stop offset="100%" stopColor="rgba(15, 23, 42, 0.1)" />
            </linearGradient>
          </defs>
          {[...Array(6)].map((_, index) => (
            <line
              key={index}
              x1={0}
              y1={(height / 5) * index}
              x2={width}
              y2={(height / 5) * index}
              stroke="rgba(148, 163, 184, 0.1)"
              strokeWidth="1"
            />
          ))}
          {Object.entries(paths).map(([band, info]) => (
            <path
              key={band}
              d={info.path}
              fill="none"
              stroke={BAND_COLORS[band]}
              strokeWidth="2"
              strokeLinejoin="round"
              strokeLinecap="round"
              opacity="0.9"
            />
          ))}
        </svg>
      </div>

      {showLegend && (
        <div className="legend">
          {DEFAULT_BANDS.map((band) => (
            <div key={band} className="legend-item">
              <span
                className="legend-swatch"
                style={{ backgroundColor: BAND_COLORS[band] }}
              />
              <span style={{ textTransform: 'capitalize' }}>{band}</span>
            </div>
          ))}
        </div>
      )}

      {showMetricGrid && latest && latest.metrics && (
        <div className="metrics-grid">
          {Object.entries(latest.metrics).map(([key, value]) => (
            <div key={key} className="metric">
              <div className="metric-label">{key.replace('_', ' ')}</div>
              <div className="metric-value">{formatNumber(value)}</div>
            </div>
          ))}
        </div>
      )}
    </div>
  );
}
