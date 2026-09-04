Here's the complete, refreshed `README.md` — everything from the original, plus today's three additions worked in cleanly.

```markdown
```
# Agentic Predictive Maintenance PoC

A multi-agent AI system that predicts industrial machine failures from real sensor
data, explains its diagnosis using retrieved historical evidence and a trained
machine learning model, recommends a maintenance action grounded in real repair
history, and knows when to act automatically versus escalate to a human engineer.

See `docs/PoC_Summary.pptx` for the executive-level presentation of this project's
architecture, results, and roadmap.

Built entirely on the [Microsoft Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
(Kaggle) — no simulated or invented data anywhere in the pipeline.

---

## Architecture

Each agent calls a dedicated tool rather than doing inline computation — the
ML/statistical logic stays separate from the agent's reasoning layer:

| # | Agent | Tool(s) it calls | What it does |
|---|---|---|---|
| 1 | Telemetry Health Agent | `predict_failure_risk` | Scores a machine's current failure risk from rolling sensor deviations |
| 2 | Prioritization Agent | `get_failure_rate_stats` | Ranks urgency using empirical MTBF-style failure statistics |
| 3 | Case Retrieval Agent | `retrieve_similar_cases` | Finds similar real historical failures via ChromaDB vector search |
| 4 | Reasoning Agent | `predict_component_failure_24h` (ML classifier) + Claude API | Diagnoses the likely failing component, combining a trained ML model's prediction with retrieved evidence |
| 5 | Recommendation Agent | `get_historical_maintenance_action` | Recommends a maintenance action based on real repair history |
| 6 | Decision & Routing Agent | (confidence threshold) | Auto-routes high-confidence cases, escalates the rest to human review |
| 7 | Feedback/Evaluation Agent | `check_against_ground_truth` | Backtests predictions against real recorded failures |

**Trained ML classifier (`predict_component_failure_24h`):** an XGBoost model
trained on 22 real engineered features (sensor telemetry, error counts, days
since maintenance, machine age/model), with a strict time-based train/test
split. Its prediction is fed directly into the Reasoning Agent's prompt as
primary evidence, and is also shown separately on the dashboard for
transparency — the Reasoning Agent explains explicitly when it defers to this
prediction over weaker evidence (e.g., the general statistical baseline).

Agents 1–6 are wired into one automated pipeline via **LangGraph** (`src/pipeline.py`).

---

## How to run it

1. Clone the repo and install dependencies, including `xgboost` (`pip install -r requirements.txt`, or see individual `pip install` commands used during development in the notebook)
2. Download the dataset via the Kaggle API into `data/raw/` (see `notebooks/01_data_exploration.ipynb`, Day 1 section)
3. Set up a `.env` file at the project root with `ANTHROPIC_API_KEY=your-key-here`
4. Run `notebooks/01_data_exploration.ipynb` from the top to build the health-scored telemetry, MTBF table, and ChromaDB Knowledge Library
5. Train the ML classifier once: `python src/tools/train_and_save_classifier.py` (saves the model to `models/`)
6. Use `src/pipeline.py`'s `build_pipeline()` to run the full agent chain on any machine

## React + FastAPI dashboard

The primary demo interface, built on the same validated pipeline — no separate logic.

1. Start the backend: `uvicorn backend_api:app --reload --port 8000`
2. Start the frontend: `cd dashboard && npm run dev`
3. Open `http://localhost:5173`

Pick a date, see machines ranked by risk:

- **Red** — the classifier predicts a real component failure within the next 24 hours (its formally validated task).
- **Yellow** — no failure predicted in the next 24 hours, but the same classifier, re-run at later time offsets, predicts one 24-72 hours out (see "Extended lookout window" under Validated Results).
- **Green** — clean across the full 72-hour lookout.

Click any machine for the full diagnosis: health score, diagnosed component,
the ML model's independent probability (shown as calibrated historical
accuracy rather than raw score for high-confidence predictions — see Known
Limitations), the AI's reasoning confidence, a grounded recommendation, and a
routing decision — each field shown with a plain-language explanation of
what it means.

**Backup interface:** `streamlit run app.py` — a simpler, single-file version of the same pipeline.

---

## Validated results

**Health scoring:** backtested on 740 real historical failures and 300
genuinely healthy periods (1,040 total cases) — **94.7% recall, 93.6%
precision**, catching failures with 24 hours of advance warning.

**ML classifier (`predict_component_failure_24h`):** trained with a strict
time-based split (train: before Nov 1 2015, test: on/after) — **0.82–0.94
recall per component** on held-out real test data. Re-confirmed live on 11
fresh real test cases (10/11 correct).

**Extended lookout window (24-72h):** the classifier's formally validated
task is a 24-hour prediction. Re-running it at later time offsets (no
retraining involved — same model, just checked further ahead) and
backtesting across 6 dates (Feb-Sep 2015, 70 real failures total) showed
recall rising from **24.3%** (24h window only) to **80.0%** (24-72h
combined), with **100% precision** (39/39) on the extended-window flags
specifically. This extrapolates beyond the classifier's originally validated
24h scope — treated as a promising but not-yet-fully-validated capability
pending a larger backtest sample, and drives the dashboard's "Yellow"
category.

**Full diagnosis chain (Reasoning Agent):** validated on multiple random
samples — 100% on a 4-case set, 80% on a fresh 10-case set, and 100% on a
5-case automated batch test through the full LangGraph pipeline, prior to
the ML classifier integration. Post-integration spot checks show the
classifier's prediction and the final diagnosis agreeing in the large
majority of cases, with disagreements explicitly explained in the reasoning.

**Recommendation grounding:** 97.6% of real historical failures have a
matching same-day maintenance record, confirming the maintenance data
genuinely reflects real repair actions rather than unrelated scheduled
servicing.

**Threshold validation:** the 0.05/0.15 High/Medium/Low health-score cutoffs
were originally chosen through iterative testing, then formally validated via
a precision-recall curve analysis. The mathematically optimal threshold
(0.032) performs only marginally better (97.6% vs 94.7% recall, 92.6% vs
93.6% precision) — current thresholds were kept for simplicity and stability.

---

## Historical failure frequency (not a 24h prediction)

The dashboard shows a secondary, honestly-labeled statistic: how often a
machine has historically failed (its own real average days-between-failures).

**This was originally built and shipped as a "24-hour failure probability"
prediction, rigorously tested, and found to have no real predictive power**
— the calculated probability was nearly identical whether checked 24 hours
before a real failure (8.02%) or on an ordinary day (7.92%); a more
sophisticated elapsed-time version also failed (15.42% vs 16.39%). Root
cause: failure timing across the 100 machines is fairly irregular
(coefficient of variation ≈ 0.67) — not on a predictable schedule.

The feature was relabeled accordingly and is now used only as a minor
tie-breaker between machines with similar health scores, never as a
day-specific prediction. **The trained ML classifier (above) is what
actually answers the 24-hour prediction question, using real sensor
data rather than failure timing alone.**

---

## What's real vs. what's a design choice

- **Real, data-derived:** health scores, MTBF statistics, retrieved case evidence, the ML classifier's predictions, maintenance recommendations, all backtest accuracy numbers.
- **Design/policy choice, not derived from data:** the confidence thresholds used for auto-routing vs. engineer review vs. urgent escalation, and the 24-72h Yellow window boundaries. These are configurable, the same way any deployed system has tunable policy parameters.

---

## Known limitations (found and documented during development)

1. **Ambiguous dual-signal cases can produce confident wrong answers.** Both the retrieval system's `dominant_sensor` matching and the ML classifier select based on the single strongest signal. When two sensors/features show comparably strong evidence at once, the system can be confidently wrong — the riskiest failure mode, since high confidence can auto-route past human review. Not yet addressed with a multi-signal weighting approach.
2. **No raw vibration/acoustic waveform data.** The dataset provides hourly aggregated sensor readings, not high-frequency time-series signal data — this PoC does not perform true NVH-style signal processing. Framed as sensor telemetry analytics.
3. **No true root-cause/mechanism data.** The dataset records which component failed, not the underlying physical cause. Framed as "likely failing component + evidence," not "root cause."
4. **Confidence thresholds are a design choice**, not derived from the data.
5. **Health score sensor weighting is currently equal, not learned.** A logistic regression on the same 1,040-case backtest showed learned weights (rotation weighted higher) improve recall to 96.5% at a small precision cost (92.2%, down from 93.6%). Investigated, documented, not yet adopted.
6. **No Memory / continuous learning loop.** The system does not currently retain engineer feedback (approve/reject decisions) or use it to improve future diagnoses — each check runs independently, with no state carried forward. A human-in-the-loop review widget exists but does not yet write outcomes back into the Knowledge Library. Identified as a genuine gap against the standard "Model / Tools / Knowledge base / Memory" agentic architecture pattern — the first three are implemented, Memory is not.
7. **No continuous monitoring or alerting.** The system is on-demand only — a user must open the dashboard and manually check. There is no automated background process that watches machines and proactively notifies anyone of emerging risk.
8. **Displayed model confidence is not calibrated to real-world accuracy.** The classifier's raw softmax output clusters heavily near 99-100%, a structural property of how tree-ensemble models combine votes rather than a sign of instability — verified directly by inspecting raw pre-softmax scores (e.g. a ~8-point internal gap becomes 99.96% vs 0.04% once exponentiated). Backtested against 300 real cases (150 real failures + 150 confirmed-healthy periods), predictions in the 99-100% confidence bucket were actually correct **95.9%** of the time, not 100%. The dashboard now displays this measured historical accuracy for high-confidence predictions rather than the raw score, to avoid overstating certainty. Lower-confidence buckets had too few samples (1-5 cases each) in this backtest to calibrate reliably and are shown as raw values with that caveat.

Additional bugs found and fixed during development (NaN validation gap, a
text-embedding search bias toward certain components, an accidental deletion
of working code during notebook cleanup, later restored and traced via
`git log -S`) are documented in the notebook's Known Issues Log.

---

## Positioning

This PoC validates the architecture and approach on a real public benchmark
dataset with backtested accuracy — not a simulated demo. A live plant pilot
would additionally need real sensor integration, site-specific criticality
weighting, validation against that plant's own failure history, and — for
genuine production use — the Memory and Monitoring/Alerting capabilities
noted above as identified but not yet implemented.
```

