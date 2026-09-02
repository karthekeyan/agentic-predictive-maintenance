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

---

## 4. Evaluation Methodology

**Backtest set:** 740 real failures (of 761; 21 excluded for insufficient rolling-window history) + 300 randomly sampled healthy periods (≥48h from any real failure, to avoid contamination).

**Metrics:** standard precision/recall via confusion matrix (TP/FP/TN/FN), computed once for the health-score tool and separately for the ML classifier.

**A rejected approach, documented for transparency:** a historical-timing-based "24h failure probability" (both a flat-rate and an elapsed-time-since-last-failure variant) was built and backtested. Both showed no meaningful separation between pre-failure and normal periods (8.02% vs. 7.92%; 15.42% vs. 16.39% average predicted probability). Root cause: failure gap coefficient of variation ≈ 0.67 across machines — too irregular for timing alone to predict a specific 24h window. Feature was retired and relabeled as a historical-frequency statistic.

---

## 5. Known Technical Limitations

- `dominant_sensor` selects only the single largest deviation; machines with two comparably strong deviations (found: comp4 vs. comp2 ambiguity) can produce a confident, incorrect classification — not yet addressed with a multi-signal weighting approach.
- Health score treats all sensors with equal (+1/+1/+1/−1) weighting. A logistic regression trained on the same 1,040-case set showed rotation carries meaningfully more signal than the others (learned weights improved recall to 96.5% at a precision cost of 1.4 points) — investigated but not yet adopted into production.
- No caching/pre-computation layer: classifier calls are made live per request; fleet-wide ranking (`/machines-at-risk`) runs the classifier only on already-flagged High/Medium machines to bound latency.
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
