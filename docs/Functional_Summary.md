# Agentic AI Predictive Maintenance — Functional Document

---

## 1. Purpose

A multi-agent AI system that monitors industrial machines, identifies which ones are showing early signs of failure, diagnoses the likely failing component, recommends a maintenance action, and decides whether to act automatically or escalate to a human engineer — built and validated entirely on real historical data, with zero simulated inputs anywhere in the pipeline.

---

## 2. Data Foundation

The system runs on the real Microsoft Azure Predictive Maintenance dataset: 100 machines, a full year of hourly sensor telemetry (vibration, voltage, pressure, rotation), real logged error events, real maintenance/repair records, and 761 real recorded component failures. No part of the system uses invented or simulated data.

---

## 3. Core Functional Components

### 3.1 Telemetry Health Agent
Calculates a real-time "health score" for any machine at any point in time, by comparing its current sensor behavior against its own historical normal. Combines all four sensors into a single number, since individual sensors alone proved to be weak, inconsistent signals — the combination is what makes the score reliable.

### 3.2 Prioritization Agent
Ranks machine urgency using empirical failure-rate statistics (Mean Time Between Failures) calculated from real historical failure gaps per machine model and component — not invented "criticality" scores.

### 3.3 Case Retrieval Agent (Knowledge Library)
Searches a database of all 761 real historical failure cases to find situations similar to what a machine is currently showing. Uses both semantic similarity and structured filtering (matching on which sensor is most abnormal) to ensure genuinely relevant matches are returned.

### 3.4 Reasoning Agent
Uses an AI language model (Claude) to produce a plain-language diagnosis of the likely failing component, grounded in the retrieved real cases, the machine-type failure-rate statistics, and (see 3.7) a trained machine learning model's prediction. Explicitly explains its reasoning, including when and why it weighs one source of evidence over another.

### 3.5 Recommendation Agent
Recommends a specific maintenance action based on what was actually done historically when similar components failed — e.g., "in 45 of 46 similar real cases, this component was replaced on the same day." Correctly handles the case where no failure is predicted, recommending no action rather than forcing an irrelevant suggestion.

### 3.6 Decision & Routing Agent
Decides, based on the confidence of the diagnosis, whether a case should go straight to the maintenance team, be flagged for engineer review, or be escalated urgently. This threshold is a configurable policy setting, kept deliberately conservative and disclosed as such.

### 3.7 Trained Machine Learning Classifier
A separately trained model (XGBoost) that independently predicts whether a machine will fail within 24 hours and which component, using 22 real engineered features from sensor, error, and maintenance data. Trained and tested with a strict time-based split (learned only from earlier data, tested only on later, unseen data) to avoid any unfair advantage. Its prediction is fed directly into the Reasoning Agent as evidence, and is also shown separately on the dashboard for transparency.

### 3.8 Feedback / Evaluation Function
Backtests the system's predictions against real recorded outcomes, producing genuine, defensible accuracy figures rather than assumed performance.

---

## 4. Validated Results

| Metric | Result | Basis |
|---|---|---|
| Recall (real failures correctly flagged) | 94.7% | 1,040 real test cases (740 real failures, 300 confirmed-healthy periods) |
| Precision (flags that were genuine) | 93.6% | Same 1,040-case test set |
| ML classifier accuracy | 82–94% recall per component | Time-based train/test split on real historical data |
| Risk-threshold optimality | Confirmed near-optimal | Formal precision-recall curve analysis |

---

## 5. Issues Found and Resolved During Development

Several real technical issues were identified through rigorous testing and corrected, rather than left unaddressed:

- **Missing/invalid sensor data** could previously produce a confident but ungrounded diagnosis — fixed with a validation guard that now correctly reports "insufficient data."
- **A systematic bias** caused the diagnosis to favor only two of four possible failing components — root-caused and fixed, improving test accuracy from 50% to 80–100%.
- **A 24-hour failure probability feature**, built using only historical failure timing, was rigorously tested and found to have no real predictive value, because real machine failures do not occur on a predictable schedule. This was proven with direct evidence, reported transparently, and the feature was honestly relabeled rather than left as a misleading claim.
- **Risk-level thresholds** were formally checked against a statistical optimization method and confirmed to already be near-optimal.

---

## 6. User-Facing Dashboard

A working web dashboard allows a user to select any date and see all 100 machines ranked by real risk. Selecting a machine shows: its health score, the diagnosed likely-failing component (or "none"), the trained ML model's independent confidence, the AI's overall reasoning confidence, a grounded maintenance recommendation, and a routing decision — each field accompanied by a plain-language explanation of what it means and how to read it.

---

## 7. Current Scope and Limitations

- Ambiguous cases where two sensors show comparably strong signals can occasionally produce a confident but incorrect diagnosis — a known, documented limitation.
- The system identifies which component is likely to fail, not the underlying physical cause of the failure.
- This is a validated proof of concept on one public dataset; deployment at a real plant would require integration with that plant's own sensors and re-validation against its own failure history.
