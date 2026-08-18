# Agentic Predictive Maintenance PoC

A multi-agent AI system that predicts industrial machine failures from real sensor
data, explains its diagnosis using retrieved historical evidence, recommends a
maintenance action grounded in real repair history, and knows when to act
automatically versus escalate to a human engineer.

Built entirely on the [Microsoft Azure Predictive Maintenance dataset](https://www.kaggle.com/datasets/arnabbiswas1/microsoft-azure-predictive-maintenance)
(Kaggle) — no simulated or invented data anywhere in the pipeline.

## Architecture

Seven agents, each calling a dedicated tool rather than doing inline computation —
the ML/statistical logic stays separate from the agent's reasoning layer:

| # | Agent | Tool it calls | What it does |
|---|---|---|---|
| 1 | Telemetry Health Agent | `predict_failure_risk` | Scores a machine's current failure risk from rolling sensor deviations |
| 2 | Prioritization Agent | `get_failure_rate_stats` | Ranks urgency using empirical MTBF-style failure statistics |
| 3 | Case Retrieval Agent | `retrieve_similar_cases` | Finds similar real historical failures via ChromaDB vector search |
| 4 | Reasoning Agent | (Claude API) | Diagnoses the likely failing component, grounded in retrieved evidence |
| 5 | Recommendation Agent | `get_historical_maintenance_action` | Recommends a maintenance action based on real repair history |
| 6 | Decision & Routing Agent | (confidence threshold) | Auto-routes high-confidence cases, escalates the rest to human review |
| 7 | Feedback/Evaluation Agent | `check_against_ground_truth` | Backtests predictions against real recorded failures |

Agents 1–6 are wired into one automated pipeline via **LangGraph** (`src/pipeline.py`).

## How to run it

1. Clone the repo and install dependencies (`pip install -r requirements.txt` — or see individual `pip install` commands used during development in the notebook)
2. Download the dataset via the Kaggle API into `data/raw/` (see `notebooks/01_data_exploration.ipynb`, Day 1 section)
3. Set up a `.env` file at the project root with `ANTHROPIC_API_KEY=your-key-here`
4. Run `notebooks/01_data_exploration.ipynb` from the top to build the health-scored telemetry, MTBF table, and ChromaDB Knowledge Library
5. Use `src/pipeline.py`'s `build_pipeline()` to run the full agent chain on any machine — see the notebook's Day 6–7 sections for example usage

## Validated results

**Health scoring (Telemetry Health Agent):** backtested on 740 real historical
failures and 300 genuinely healthy periods (1,040 total cases) —
**94.7% recall, 93.6% precision**, catching failures with 24 hours of advance
warning.

**Full diagnosis chain (Reasoning Agent):** validated on multiple random samples
after fixing a retrieval bug (see Known Limitations) — 100% on a 4-case set,
80% on a fresh 10-case set, and 100% on a 5-case automated batch test through
the full LangGraph pipeline.

**Recommendation grounding:** 97.6% of real historical failures have a matching
same-day maintenance record, confirming the maintenance data genuinely reflects
real repair actions rather than unrelated scheduled servicing.

## What's real vs. what's a design choice

- **Real, data-derived:** health scores, MTBF statistics, retrieved case evidence, maintenance recommendations, backtest accuracy numbers.
- **Design/policy choice, not derived from data:** the confidence thresholds used for auto-routing (70%) vs. engineer review (50–70%) vs. urgent escalation (<50%). These are configurable, the same way any deployed system has tunable policy parameters.

## Known limitations (found and documented during development)

1. **Ambiguous dual-signal cases can produce confident wrong answers.** The
   system identifies a machine's "dominant sensor" to match against historical
   cases. When two sensors show comparably strong deviations at once (e.g.,
   Machine 7: vibration 20.3% vs. rotation -21.3%), the system can pick the
   wrong one with high confidence — the riskiest failure mode, since high
   confidence auto-routes past human review. Not yet fixed; a priority item
   for future work (e.g., considering the top-2 dominant sensors when they are
   close in magnitude, not just the single strongest).
2. **No raw vibration/acoustic waveform data.** The dataset provides hourly
   aggregated sensor readings, not high-frequency time-series signal data — so
   this PoC does not perform true NVH-style signal processing, despite that
   being referenced in earlier project materials. Framed here as sensor
   telemetry analytics.
3. **No true root-cause/mechanism data.** The dataset records which component
   failed, not the underlying physical cause (e.g., not "bearing wear from
   misalignment"). The Reasoning Agent's output is framed as "likely failing
   component + evidence," not "root cause," to stay within what's provable.
4. **Confidence thresholds are a design choice**, not derived from the data
   (see above) — worth stating explicitly in any demo or pitch.

Two additional bugs were found and fixed during development (NaN validation
gap, and a text-embedding search bias toward certain components) — see the
notebook's Known Issues Log for full detail on both, including root cause and
verification.

## Positioning

This PoC validates the architecture and approach on a real public benchmark
dataset with backtested accuracy — not a simulated demo. A live plant pilot
would additionally need real sensor integration, site-specific criticality
weighting, and validation against that plant's own failure history.
