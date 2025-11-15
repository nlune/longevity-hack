import LiveWaves from './components/LiveWaves.jsx';

const WS_URL = import.meta.env.VITE_WS_URL || 'ws://localhost:8000/ws/metrics';

export default function App() {
  return (
    <div className="app-shell">
      <div className="card">
        <h1>Muse Metrics Stream</h1>
        <p>Live band powers and derived relaxation/concentration metrics from the FastAPI backend.</p>
        <LiveWaves url={WS_URL} />
      </div>
    </div>
  );
}
