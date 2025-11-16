# Stress Compass

Stress Compass helps knowledge workers notice and neutralize chronic stress patterns during the workday. High stress is strongly associated with hypertension, insulin resistance, depression, and immune dysfunction—conditions that can trim 5–10 years off of life expectancy. Most of us spend the majority of our waking hours at a desk, barely registering when our nervous system tips into fight-or-flight. By combining Muse EEG ratios with HRV-derived indicators, Stress Compass builds a personal baseline and continuously highlights deviations that signal mental overload.

The goal is simple: deliver a subtle nudge the moment your mind drifts into dysregulation, so you can take a breathing break, stand up, or reframe your workload before stress compounds. Over time these micro-interventions reduce allostatic load, improving wellbeing and productivity while protecting long-term cognitive and cardiovascular health.

## Getting Started

### Backend (FastAPI)

1. Install dependencies (`uv sync` or `pip install -e .`).
2. Start the server with `python main.py` (or `uvicorn backend:app --reload`).
3. `POST /baseline` records a one-minute calibration window (EEG bands + HRV).
4. `POST /monitor` samples a fresh window, compares against baseline, and returns deviations plus alerts when stress proxies spike.
5. `POST /notify` (macOS) schedules a desktop notification with a configurable delay—used now for manual tests and later by the agentic workflow.
6. `GET /calendar/events` & `POST /agent/mock-trigger` integrate with Google Calendar and the Groq-based agent: the agent inspects the current/next meetings, decides when to nudge, and sends the intervention via `/notify`.
7. `GET /metrics` and `ws://localhost:8000/ws/metrics` expose the raw streaming data for the frontend.

_Pro tip:_ When no Muse headset or PPG stream is detected, set `MUSE_MOCK_MODE=true` to enable the built-in simulator for UI testing.

### Frontend (Vite + React)

1. `cd frontend && npm install`
2. `npm run dev` (defaults to http://localhost:5173)
3. Configure the backend locations via `.env` (`VITE_API_URL`, `VITE_WS_URL`) if you’re not on localhost.
4. The app walks you through baseline capture, shows live band waves, plots deviation history, and surfaces relaxation/engagement/stress gauges alongside desktop notifications for negative events.
5. Enable browser notifications to receive self-regulation prompts even when the tab is in the background.

### How It Works

- **Baseline calibration:** A one-minute session measures personal EEG ratios (alpha/theta/beta) and HRV (RMSSD, SDNN, LF/HF). The system stores means and standard deviations for each metric.
- **Deviation monitoring:** Every ~30 seconds the backend collects a fresh window, computes z-scores relative to baseline, and aggregates them into three interpretable proxies:
  - *Relaxation* blends alpha/theta ratios with HRV RMSSD (higher = calmer, parasympathetic state).
  - *Engagement* balances beta concentration against theta drift (higher = focused, mentally present).
  - *Stress / scatter* weighs LF/HF against HRV RMSSD to detect sympathetic spikes.
  A weighted composite of all deviations (positive spikes inverted, negative drops preserved) becomes the **stress index**, the single number used for current alerts and future agentic logic.
- **Nudges, not nags:** Positive deviations (e.g., calmer-than-usual alpha or steadier HRV) appear in green cards and history charts. Negative deviations (stress spikes, engagement drops, HRV dips) tint red/orange and may trigger notifications so you can intervene early.
- **Agentic trigger:** When the stress index crosses a configurable threshold (default 1.5σ) the backend agent automatically checks Google Calendar (current/next meetings) and crafts an intervention via Groq. If you’re in a meeting the notification can be delayed; otherwise it fires immediately. A manual mock trigger remains available for demos.

### Calendar & Agent Integration

1. Create a Google Cloud project, enable the Calendar API, and configure an OAuth **Web application** client (authorized redirect URI: `http://localhost:8000/calendar/oauth/callback`).
2. Set environment variables:
   - `GOOGLE_OAUTH_CLIENT_ID=...`
   - `GOOGLE_OAUTH_CLIENT_SECRET=...`
   - Optional: `GOOGLE_OAUTH_REDIRECT_URI` (defaults to `http://localhost:8000/calendar/oauth/callback`).
   - Optional: `GOOGLE_CALENDAR_ID` (defaults to `primary`).
   - Optional: `GOOGLE_TOKEN_STORE` to change where OAuth tokens are cached locally.
3. Provide a `GROQ_API_KEY` so the agent can craft contextual interventions.
4. In the UI, click **Connect calendar**. A Google consent window opens; select your calendar account. Once connected, you’ll see today’s events listed and the agent can reason about current/next meetings.
5. Use **Mock stress trigger** at the bottom of the UI only if you need to simulate a high stress index manually. Normally the agent fires automatically whenever the live stress index crosses the threshold. The backend agent checks current/next events, decides whether to delay the notification (e.g., during a meeting), and schedules the desktop intervention via `/notify`.

By cycling between awareness and action throughout the day, Stress Compass helps dial down chronic sympathetic load—the exact pattern linked to burnout, anxiety, and metabolic disease—supporting a longer, healthier life.
