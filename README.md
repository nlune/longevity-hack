## Berlin Longevity Hack

Install dependencies.
Install muselsl, then in a terminal run `muselsl stream` (make sure headset is on).
Run `python main.py` in another terminal for backend.

### FastAPI backend

1. Install dependencies (via `uv sync` or `pip install -e .`).
2. Start the server with `python main.py` (or `uvicorn backend:app --reload`).
3. `GET /metrics` returns a snapshot of the Muse band powers and derived metrics.
4. Connect to `ws://localhost:8000/ws/metrics` to receive a live JSON stream for the frontend.

### React frontend demo

1. `cd frontend && npm install` to install dependencies.
2. Start the dev server with `npm run dev` (defaults to http://localhost:5173).
3. The UI connects to `ws://localhost:8000/ws/metrics` by default; override by exporting `VITE_WS_URL`.
4. Open the Vite dev server in the browser to view the live multi-band wave plot and metrics.
5. Make sure your laptop doesn't stop the notifications through some configurable-means.

**Tip:** When no Muse headset is detected, the backend automatically falls back to synthetic EEG data so you can test locally. Set `MUSE_MOCK_MODE=false` to disable the mock stream or `MUSE_MOCK_MODE=true` to force it.
