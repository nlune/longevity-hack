import { useMemo } from 'react';
import { computeMindStateScores } from '../utils/mindState.js';

const SERIES = [
  { key: 'hrvComposite', label: 'HRV composite', color: '#22c55e' },
  { key: 'relaxation', label: 'Relaxation proxy', color: '#0ea5e9' },
  { key: 'engagement', label: 'Engagement proxy', color: '#6366f1' },
  { key: 'stress', label: 'Stress proxy', color: '#f97316' },
];

function clampZ(value) {
  if (value === undefined || value === null || Number.isNaN(value)) {
    return 0;
  }
  return Math.max(-3, Math.min(3, value));
}

function formatTimeLabel(timestamp) {
  if (!timestamp) {
    return '';
  }
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' });
}

export default function HRVDeviationChart({ history = [], width = 720, height = 220 }) {
  const { paths, zeroY, ticks, latestValues, timeRangeMinutes } = useMemo(() => {
    if (!history.length) {
      return { paths: [], zeroY: height / 2, ticks: [], latestValues: {}, timeRangeMinutes: 0 };
    }

    const minTimestamp = history[0]?.timestamp;
    const maxTimestamp = history[history.length - 1]?.timestamp;
    const timeRangeMinutes = Math.max(
      0,
      ((maxTimestamp ?? 0) - (minTimestamp ?? 0)) / 60
    );

    const valueToY = (value) => {
      const normalized = (clampZ(value) + 3) / 6; // 0..1
      return height - normalized * height;
    };

    const step = history.length > 1 ? width / (history.length - 1) : width;

    const paths = SERIES.map(({ key, ...rest }) => {
      const commands = history.map((point, index) => {
        const scores = computeMindStateScores(point?.deviations || {});
        const deviation =
          key === 'hrvComposite'
            ? scores.hrvComposite
            : scores[key];
        const x = index * step;
        const y = valueToY(deviation * 3);
        return `${index === 0 ? 'M' : 'L'} ${x.toFixed(2)} ${y.toFixed(2)}`;
      });
      return { key, path: commands.join(' '), ...rest };
    });

    const tickCount = 4;
    const ticks = new Array(tickCount).fill(0).map((_, idx) => {
      const fraction = idx / (tickCount - 1);
      const timestamp = (minTimestamp || 0) + fraction * ((maxTimestamp || 0) - (minTimestamp || 0));
      const x = fraction * width;
      return { x, label: formatTimeLabel(timestamp) };
    });

    const latestDeviation = history[history.length - 1]?.deviations ?? {};
    const latestScores = computeMindStateScores(latestDeviation);
    const latestValues = {
      hrvComposite: latestScores.hrvComposite,
      relaxation: latestScores.relaxation,
      engagement: latestScores.engagement,
      stress: latestScores.stress,
    };

    return { paths, zeroY: valueToY(0), ticks, latestValues, timeRangeMinutes };
  }, [history, height, width]);

  if (!history.length) {
    return (
      <div className="chart-placeholder">
        <p className="muted">Deviation trends will appear once monitoring begins.</p>
      </div>
    );
  }

  const windowLabel = timeRangeMinutes >= 1
    ? `${Math.round(timeRangeMinutes)} min`
    : `${Math.max(1, Math.round(timeRangeMinutes * 60))} sec`;

  return (
    <div className="hrv-chart" role="img" aria-label="HRV deviations over time">
      <svg viewBox={`0 0 ${width} ${height}`} className="hrv-chart-svg">
        <defs>
          <linearGradient id="hrvGrid" x1="0" y1="0" x2="0" y2="1">
            <stop offset="0%" stopColor="rgba(148, 163, 184, 0.15)" />
            <stop offset="100%" stopColor="rgba(15, 23, 42, 0.05)" />
          </linearGradient>
        </defs>
        {[...Array(5)].map((_, index) => (
          <line
            key={`grid-${index}`}
            x1={0}
            x2={width}
            y1={(height / 4) * index}
            y2={(height / 4) * index}
            stroke="rgba(148, 163, 184, 0.2)"
            strokeWidth={1}
          />
        ))}
        <line
          x1={0}
          y1={zeroY}
          x2={width}
          y2={zeroY}
          stroke="rgba(248, 250, 252, 0.4)"
          strokeDasharray="4 4"
        />
        {paths.map(({ key, path, color }) => (
          <path
            key={key}
            d={path}
            fill="none"
            stroke={color}
            strokeWidth={2.5}
            strokeLinejoin="round"
            strokeLinecap="round"
          />
        ))}
      </svg>
      <div className="hrv-chart-footer">
        <div className="hrv-legend">
          {SERIES.map((series) => (
            <div key={series.key} className="legend-item">
              <span className="legend-swatch" style={{ backgroundColor: series.color }} />
              <span>{series.label}</span>
              <span className="legend-value">
                {latestValues[series.key] !== undefined
                  ? clampZ(latestValues[series.key] * 3).toFixed(1)
                  : '--'}
                σ
              </span>
            </div>
          ))}
        </div>
        <div className="hrv-ticks">
          {ticks.map((tick, index) => (
            <span key={`tick-${index}`} style={{ left: `${tick.x}px` }}>
              {tick.label}
            </span>
          ))}
        </div>
        <div className="hrv-window-label">{windowLabel} window</div>
      </div>
    </div>
  );
}
