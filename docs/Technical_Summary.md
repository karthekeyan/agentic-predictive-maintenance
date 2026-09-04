# Agentic AI Predictive Maintenance — Technical Summary

---

## 1. Architecture Overview

**Pattern:** agents reason, tools compute. Every ML/statistical calculation lives in a standalone tool function; each agent wraps a tool and adds a decision layer on top. This keeps the LLM's role limited to reasoning and explanation — it never replaces the underlying predictive models.

**Orchestration:** LangGraph wires all agents into a single automated state-machine pipeline (`src/pipeline.py`), with a shared `PipelineState` object passed between nodes.

```
Telemetry Health → Prioritization → Case Retrieval → Reasoning
    (calls classifier internally) → Recommendation → Decision & Routing
```

---

## 2. Technology Stack

| Layer | Technology |
|---|---|
| Orchestration | LangGraph |
| Reasoning LLM | Claude (Anthropic API) |
| Vector store / RAG | ChromaDB |
| ML classifier | XGBoost (scikit-learn ecosystem) |
| Data processing | pandas, NumPy |
| Backend API | FastAPI |
| Frontend | React + Vite |
| Backup UI | Streamlit |
| Version control | Git / GitHub |

---

## 3. Core Algorithms

### 3.1 Health Score
```
deviation(sensor) = (rolling_24h_avg − machine_own_historical_avg) / machine_own_historical_avg
health_score = vibration_dev + volt_dev + pressure_dev − rotate_dev
```
Rolling 24h window smooths hourly sensor noise. Deviation is computed against each machine's own historical average (not a fleet-wide baseline), since machines vary naturally by design. Rotation is subtracted because it trends downward before failure, while the other three trend upward — validated via lead-time analysis across 761 real failures.

**Risk thresholds:** Low < 0.05 ≤ Medium < 0.15 ≤ High. Empirically chosen, later confirmed near-optimal via a precision-recall curve on 1,040 real test cases (optimal threshold: 0.0318 vs. current 0.05 — a ~1–3 point difference in precision/recall, not adopted due to marginal gain).

### 3.2 MTBF (Prioritization)
Real historical failure gaps computed via `.diff()` on sorted per-model, per-component failure timestamps — not a fleet-wide average, avoiding a bug found early where multiple machines returned identical statistics.

### 3.3 Retrieval (RAG)
ChromaDB vector store of 761 real failure cases, each embedded from a natural-language description of that case's real sensor deviations. Retrieval combines:
- **Structured filter** on `dominant_sensor` (the single largest deviation, computed from real numbers) — added after discovering pure semantic search could not distinguish numeric magnitude (e.g., "voltage 12%" vs "voltage 1%" read as similar text)
- **Semantic similarity** ranking within the filtered set
- Dominant-sensor matches are prioritized in the merge and never outranked by broader, less-reliable full-text matches (a second bug found and fixed)

### 3.4 ML Classifier (24h component failure prediction)
XGBoost multi-class classifier (`comp1`–`comp4`, `none`). 22 features:
- 3-hour bucketed telemetry min/max (`{sensor}_{min,max}_3h`)
- Rolling 24-hour one-hot error counts (`errorID_{id}_24h`)
- Days since last maintenance per component (`days_since_comp{1-4}`, backward `merge_asof`)
- Machine age + one-hot model

**Label construction:** forward `merge_asof` to the next real failure; labeled with that failure's component if ≤24 hours away, else `none`.

**Split:** strict time-based — train on data before 2015-11-01, test on data at/after — no shuffling, avoiding temporal leakage.

**Validated recall:** 0.82 (comp1) – 0.94 (comp4) on held-out test data. Re-confirmed live on 11 fresh real cases (10/11 correct).

**Inference:** single-machine/single-date feature builder (`build_features_for_prediction.py`) mirrors the batch training logic exactly, using only data at or before the query timestamp.

**Extended-window inference:** the same trained classifier (no retraining) is additionally called at `as_of + 24h` and `as_of + 48h` for any machine not flagged in the standard 24h window, to check whether a failure becomes predictable further out. This is a pure inference-time extension — the model's training, features, and label definition are unchanged; only the query timestamp shifts. See Section 4 for backtested accuracy of this extension.

### 3.5 Confidence Output (Softmax)
`predict_proba()` returns class probabilities computed via softmax over the model's raw per-class margin scores (`predict(output_margin=True)`). Verified directly against a real prediction: raw margin gaps of roughly 8 points between the top two classes produced a post-softmax split of 99.96% vs. 0.04% — confirming that softmax's exponential term (`e^x`) sharply amplifies whichever class is already ahead, rather than the model being unusually decisive at the margin-score level. See Section 4 for the calibration backtest quantifying how this raw score relates to actual accuracy.

---

## 4. Evaluation Methodology

**Backtest set:** 740 real failures (of 761; 21 excluded for insufficient rolling-window history) + 300 randomly sampled healthy periods (≥48h from any real failure, to avoid contamination).

**Metrics:** standard precision/recall via confusion matrix (TP/FP/TN/FN), computed once for the health-score tool and separately for the ML classifier.

**A rejected approach, documented for transparency:** a historical-timing-based "24h failure probability" (both a flat-rate and an elapsed-time-since-last-failure variant) was built and backtested. Both showed no meaningful separation between pre-failure and normal periods (8.02% vs. 7.92%; 15.42% vs. 16.39% average predicted probability). Root cause: failure gap coefficient of variation ≈ 0.67 across machines — too irregular for timing alone to predict a specific 24h window. Feature was retired and relabeled as a historical-frequency statistic.

**Confidence calibration backtest:** 300 real cases (150 real failures, `as_of` = failure time − 12h; 150 confirmed-healthy periods, sampled ≥48h from any real failure) were run through the classifier, bucketing `predict_proba` output against actual correctness. Result: the 99–100% confidence bucket (291 of 300 cases) showed 95.9% actual accuracy — a real, quantified overconfidence gap, not a broken model. Buckets below 99% had too few samples (1–5 cases each) to be statistically meaningful and are not used for calibrated display.

**Extended-window (24-72h) backtest:** run across 6 dates (2015-02-01, 03-15, 05-01, 06-03, 07-15, 09-01) × 100 machines, comparing real recorded failures within 96h of each test date against classifier output at `as_of`, `as_of+24h`, and `as_of+48h`. Aggregate across all 6 dates (70 real failures total): standard 24h-window recall 24.3% (17/70), extended-window recall 80.0% (56/70), extended-window flag precision 100% (39/39). Per-date recall ranged 69.2%–90.0%, indicating the effect is consistent rather than date-specific. A separate lead-time trajectory check (9 sampled real failures, confidence sampled at 72/48/24/12/6/2h before failure) showed confidence rising from a mean of ~0.001% (72h, 48h) to ~99%+ (24h and closer) — a step change rather than a gradual gradient, consistent with the softmax amplification behavior noted in Section 3.5.

---

## 5. Known Technical Limitations

- `dominant_sensor` selects only the single largest deviation; machines with two comparably strong deviations (found: comp4 vs. comp2 ambiguity) can produce a confident, incorrect classification — not yet addressed with a multi-signal weighting approach.
- Health score treats all sensors with equal (+1/+1/+1/−1) weighting. A logistic regression trained on the same 1,040-case set showed rotation carries meaningfully more signal than the others (learned weights improved recall to 96.5% at a precision cost of 1.4 points) — investigated but not yet adopted into production.
- Displayed classifier confidence is calibrated only for the 99–100% range (95.9% real accuracy per Section 4); lower confidence values are shown raw, without a calibration claim, due to insufficient backtest sample size in those ranges.
- The 24-72h extended-window logic queries the classifier beyond its trained/validated 24h task boundary. Backtested performance is strong (Section 4) but based on a limited sample (70 real failures, 6 dates) — treated as a promising, not yet fully validated, capability.
- No caching/pre-computation layer: classifier calls are made live per request. Fleet-wide ranking (`/machines-at-risk`) now runs the classifier on **every** machine for the standard 24h window (previously limited to already-flagged machines, changed to guarantee no real failure is missed), plus up to two additional calls per machine (24-48h, 48-72h) for machines not flagged in the standard window — up to ~3x the classifier calls of the original implementation, increasing endpoint latency accordingly.
- No authentication, rate limiting, or multi-tenant data isolation on the FastAPI backend — acceptable for a local PoC, not for external deployment.
- No structured logging/tracing (e.g., LangSmith) across agent runs; debugging currently relies on manual notebook re-execution.

---

## 6. Repository Structure

```
src/
├── agents/          — reasoning layer (7 agents)
├── tools/           — pure computation (ML models, stats, retrieval, feature engineering)
notebooks/           — exploration, validation, backtesting
dashboard/           — React frontend
backend_api.py       — FastAPI service layer
pipeline.py          — LangGraph orchestration
models/              — trained classifier artifacts (.pkl, feature_columns.json)
```