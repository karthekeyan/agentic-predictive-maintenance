# Agentic AI Predictive Maintenance — User Guide

A step-by-step guide to starting, using, and interpreting the predictive maintenance dashboard.

---

## 1. Starting the Application

You need **two things running at the same time**, in two separate Command Prompt windows, plus a browser tab.

### Terminal 1 — Backend

```
cd D:\Karthikeyan\CoBuildX\PoC\agentic-predictive-maintenance
uvicorn backend_api:app --reload --port 8000
```
Wait for the message `Ready.` to appear before continuing.

### Terminal 2 — Frontend

Open a **second, separate** Command Prompt window:
```
cd D:\Karthikeyan\CoBuildX\PoC\agentic-predictive-maintenance\dashboard
npm run dev
```
Wait for `Local: http://localhost:5173/` to appear.

### Open the dashboard

In a browser, go to:
```
http://localhost:5173
```

**Important:** both terminal windows must stay open the entire time you're using the dashboard. Closing either one will break the app.

### Stopping the application

In each terminal window, press **Ctrl + C**.

---

## 2. Using the Dashboard

### Step 1 — Pick a date

Use the **Select date** field at the top and click **Check machines**. This shows every machine's condition **as of that specific date** — the system only uses data available up to that point, exactly as it would in real use.

**Note:** this check now runs the ML classifier at up to three different time windows per machine (see "How the list is ordered" below), so it may take a little longer to load than a simple single-pass check.

### Step 2 — Read the machine list

Each row shows:

| Element | Meaning |
|---|---|
| Machine name | e.g., "Machine 66" |
| Health Risk badge | High (red), Medium (amber), or Low (green) |
| "fails ~every Xd" | A secondary, historical statistic — how often this machine has tended to fail in the past |

**How the list is ordered — Red / Yellow / Green explained:**

- **Red (High):** the ML classifier predicts a real component failure within the **next 24 hours** — its formally validated task. Ranked by the model's own probability.
- **Yellow (Medium):** no failure predicted in the next 24 hours, but the *same* classifier — re-checked 24 to 72 hours further out — predicts one in that window. This extends the model beyond its originally validated 24-hour scope; backtesting across 6 historical dates showed this catches significantly more real failures (recall rising from 24.3% to 80.0%) with no false alarms observed in that test (100% precision, 39/39).
- **Green (Low):** clean across the full 72-hour lookout — kept sorted by health score.

### Step 3 — Click a machine for the full diagnosis

Clicking any machine runs the complete pipeline and shows five cards:

| Card | What it tells you |
|---|---|
| **Health Score** | The exact number behind the risk badge. Closer to 0 is normal; higher means more unusual. |
| **Diagnosis** | The specific component most likely to fail in the next 24 hours (or "None"). |
| **Probability** | How sure the trained ML model is about this specific prediction — see Section 4 for how to read this number. |
| **Reasoning Confidence** | How confident the overall diagnosis is, after weighing the ML model, similar past cases, and general statistics together. |
| **Routing** | What happens next — see below. |

Below the cards, two boxes explain the decision in plain language:
- **Reasoning** — why the system reached this diagnosis, including which evidence it trusted most
- **Recommendation** — a specific maintenance action, based on what was actually done historically for similar real cases

---

## 3. Understanding "Routing"

| Value | What it means |
|---|---|
| **Auto Route** (or "No Action" if Diagnosis is None) | Confidence is high enough to act without a human check |
| **Engineer Review** | Moderate confidence — a human should verify before acting |
| **Escalate Urgent** | Low or no confidence — needs immediate human attention |

---

## 4. Reading a Result — Worked Example

**Machine 16, Diagnosis: Comp2, Probability: 99.96%, Reasoning Confidence: 85%, Routing: Auto Route**

This means: the trained ML model is essentially certain component 2 is the one likely to fail, the overall reasoning (which also checked similar real past cases) largely agrees, and the system is confident enough to route this straight to the maintenance team without needing a human to double-check first.

**Why Probability is almost always a very high number (often 99%+):** this is a real, verified characteristic of the underlying model, not a display error — see the FAQ below for the full explanation.

**If Probability and Diagnosis ever seem to disagree** (e.g., Diagnosis says one component but Probability's underlying prediction was for a different one), the Reasoning box will explain why — this is intentional, not a bug; it means the AI weighed conflicting evidence and made a judgment call.

---

## 5. Frequently Asked Questions

**Why is Probability almost always 99%+, sometimes exactly 100%?**
This is a genuine, measured characteristic of the trained model, not a bug. The model (XGBoost) combines many small internal votes into one score, and when those votes largely agree — which happens often with this sensor data — the math used to convert that into a percentage sharply amplifies the lead, producing near-100% even when the underlying signal wasn't overwhelmingly one-sided. We tested this directly: in a 300-case backtest, predictions the model called "99-100% confident" were actually correct **95.9%** of the time — high, but not literally 100%. For that reason, the dashboard shows this measured historical accuracy for high-confidence predictions, rather than the raw score.

**Why does the dashboard take a moment to load after clicking a machine?**
The full diagnosis runs several real steps live — a database search, a trained ML model, and a call to an AI reasoning model — this takes a few seconds, unlike the initial machine list which is faster.

**Why do some machines show "None" as the diagnosis?**
This means the system does not expect a failure in the next 24 hours for that machine — a genuine, valid, positive result, not a missing answer.

**Can I trust the numbers shown?**
The health score and the ML classifier have both been rigorously backtested against real historical outcomes (94.7% recall / 93.6% precision for health scoring; 82–94% recall per component for the ML classifier, 24.3%→80.0% recall for the extended Yellow window). The "fails ~every Xd" statistic is a weaker, secondary signal — it should not be read as a prediction.

**What if the page shows an error or won't load?**
Confirm both terminal windows are still running without errors. If the backend terminal shows a red error message, that's usually the actual cause — check what it says before troubleshooting the browser.

---

## 6. Known Limitations (Read Before Presenting to Others)

- Machines showing two comparably strong abnormal signals at once can occasionally produce a confident but incorrect diagnosis.
- The system identifies *which* component is likely to fail, not the underlying physical cause.
- The displayed model confidence, while historically well-calibrated at the 99-100% range (95.9% real accuracy, based on 291 backtested cases), has too few backtested examples in the mid-confidence range to calibrate reliably there — those values are shown as raw scores with that caveat.
- The Yellow (24-72h) category extends the classifier beyond its originally validated 24-hour task. Backtesting across 6 dates was strongly positive (80.0% recall, 100% precision) but this should be treated as promising evidence, not yet a fully validated production capability — a larger, ongoing backtest is recommended before broader roll-out.
- This is a validated proof of concept on one public dataset — not yet connected to a real plant's live sensors.
- The system does not currently monitor automatically or send alerts — you must open the dashboard and check manually.
- The system does not yet learn from engineer feedback over time (the Approve/Reject buttons do not yet feed back into future diagnoses).

For full technical and business detail, see the companion `Technical_Summary.md`, `Functional_Summary.md`, and `Business_Summary.md` documents, or `docs/PoC_Summary.pptx`.