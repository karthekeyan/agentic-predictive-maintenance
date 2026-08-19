# How to Start the Application

Two versions exist. Use whichever fits the moment — React for a polished demo, Streamlit as a quick backup.

---

## Option A: React Dashboard (polished, client-facing)

Needs **two terminal windows open at the same time**, plus a browser tab.

### Terminal 1 — Backend (FastAPI)
```
cd D:\Karthikeyan\CoBuildX\PoC\agentic-predictive-maintenance
uvicorn backend_api:app --reload --port 8000
```
Wait for `Ready.` to appear before moving on.

### Terminal 2 — Frontend (React)
```
cd D:\Karthikeyan\CoBuildX\PoC\agentic-predictive-maintenance\dashboard
npm run dev
```
Wait for `Local: http://localhost:5173/` to appear.

### Browser
Open: **http://localhost:5173**

**Important:** both terminals must stay open and running the whole time you're using the dashboard. Closing either one breaks the app (the backend serves the data; the frontend just displays it).

---

## Option B: Streamlit (simple, single-window backup)

Needs **one terminal window**. Opens the browser automatically.

```
cd D:\Karthikeyan\CoBuildX\PoC\agentic-predictive-maintenance
streamlit run app.py
```
Opens automatically at **http://localhost:8501**

---

## To stop either version

In each terminal window that's running something: press **Ctrl + C**.

---

## Quick troubleshooting

| Symptom | Likely cause |
|---|---|
| React dashboard loads but machine list stays empty, or console shows `ERR_CONNECTION_REFUSED` | Backend (Terminal 1) isn't running — start it |
| Console shows a CORS error | Backend isn't allowing the frontend's port — already fixed once (5173 added), shouldn't recur unless the dashboard's port changes |
| `'uvicorn' is not recognized` | Run `pip install fastapi uvicorn` first |
| `'npm' is not recognized` | Node.js isn't installed — see Day 7 chat history for install steps |
| Streamlit shows a stale/wrong result for every machine | Restart Streamlit fully (Ctrl+C, then rerun) — module caching can serve old code otherwise |

---

## Known open item (not yet fixed)

The React dashboard's `/diagnose` endpoint has shown at least one case (Machine 36) where retrieval returned no similar historical cases, causing the Reasoning Agent to fall back to a low-confidence statistical guess — different from what the notebook testing showed for similar machines. Not yet root-caused; flagged to investigate before calling the React dashboard fully validated.