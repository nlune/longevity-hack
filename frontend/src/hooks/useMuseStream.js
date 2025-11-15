import { useEffect, useMemo, useRef, useState } from 'react';

export const DEFAULT_BANDS = ['delta', 'theta', 'alpha', 'beta'];

export function useMuseStream(url, { maxPoints = 180 } = {}) {
  const [points, setPoints] = useState([]);
  const [status, setStatus] = useState('connecting');
  const [error, setError] = useState(null);
  const socketRef = useRef(null);

  useEffect(() => {
    const socket = new WebSocket(url);
    socketRef.current = socket;

    socket.onopen = () => setStatus('online');
    socket.onclose = () => setStatus('disconnected');
    socket.onerror = () => {
      setError('Failed to connect to backend WebSocket.');
      setStatus('error');
    };
    socket.onmessage = (event) => {
      try {
        const payload = JSON.parse(event.data);
        if (payload?.error) {
          setError(payload.error);
          return;
        }
        if (payload && payload.bands) {
          setPoints((prev) => {
            const nextPoints = [...prev, payload];
            if (nextPoints.length > maxPoints) {
              nextPoints.splice(0, nextPoints.length - maxPoints);
            }
            return nextPoints;
          });
          setError(null);
        }
      } catch (err) {
        console.error('Failed to parse message', err);
      }
    };

    return () => {
      socket.close();
    };
  }, [url, maxPoints]);

  const latest = points[points.length - 1];
  const bandSeries = useMemo(() => {
    const series = {};
    DEFAULT_BANDS.forEach((band) => {
      series[band] = points.map((point) => ({
        timestamp: point.timestamp,
        value: point.bands[band],
      }));
    });
    return series;
  }, [points]);

  return { points, latest, bandSeries, status, error };
}
