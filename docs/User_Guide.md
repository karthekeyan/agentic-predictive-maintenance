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

### Step 2 — Read the machine list

Each row shows:

| Element | Meaning |
|---|---|
| Machine name | e.g., "Machine 66" |
| Health Risk badge | High (red), Medium (amber), or Low (green) — how unusual this machine's sensors look right now |
| "fails ~every Xd" | A secondary, historical statistic — how often this machine has tended to fail in the past |

**How the list is ordered:** High and Medium risk machines are ranked using the real trained ML model's failure probability (the most accurate signal available). Low risk machines are kept sorted by health score underneath.

### Step 3 — Click a machine for the full diagnosis

Clicking any machine runs the complete pipeline and shows five cards:

| Card | What it tells you |
|---|---|
| **Health Score** | The exact number behind the risk badge. Closer to 0 is normal; higher means more unusual. |
| **Diagnosis** | The specific component most likely to fail in the next 24 hours (or "None"). |
| **Probability** | How sure the trained ML model is about this specific prediction. |
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

**Machine 16, Diagnosis: Comp2, Probability: 100.0%, Reasoning Confidence: 85%, Routing: Auto Route**

This means: the trained ML model is essentially certain component 2 is the one likely to fail, the overall reasoning (which also checked similar real past cases) largely agrees, and the system is confident enough to route this straight to the maintenance team without needing a human to double-check first.

**If Probability and Diagnosis ever seem to disagree** (e.g., Diagnosis says one component but Probability's underlying prediction was for a different one), the Reasoning box will explain why — this is intentional, not a bug; it means the AI weighed conflicting evidence and made a judgment call.

---

## 5. Frequently Asked Questions

**Why does the dashboard take a moment to load after clicking a machine?**
The full diagnosis runs several real steps live — a database search, a trained ML model, and a call to an AI reasoning model — this takes a few seconds, unlike the initial machine list which is faster.

**Why do some machines show "None" as the diagnosis?**
This means the system does not expect a failure in the next 24 hours for that machine — a genuine, valid, positive result, not a missing answer.

**Can I trust the numbers shown?**
The health score and the ML classifier have both been rigorously backtested against real historical outcomes (94.7% recall / 93.6% precision for health scoring; 82–94% recall per component for the ML classifier). The "fails ~every Xd" statistic is a weaker, secondary signal — it should not be read as a prediction.

**What if the page shows an error or won't load?**
Confirm both terminal windows are still running without errors. If the backend terminal shows a red error message, that's usually the actual cause — check what it says before troubleshooting the browser.

---

## 6. Known Limitations (Read Before Presenting to Others)

- Machines showing two comparably strong abnormal signals at once can occasionally produce a confident but incorrect diagnosis.
- The system identifies *which* component is likely to fail, not the underlying physical cause.
- This is a validated proof of concept on one public dataset — not yet connected to a real plant's live sensors.
- The system does not currently monitor automatically or send alerts — you must open the dashboard and check manually.
- The system does not yet learn from engineer feedback over time (the Approve/Reject buttons do not yet feed back into future diagnoses).

For full technical and business detail, see the companion `Technical_Summary.md`, `Functional_Summary.md`, and `Business_Summary.md` documents, or `docs/PoC_Summary.pptx`.
