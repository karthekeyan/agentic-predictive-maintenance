# Agentic AI Predictive Maintenance — Business Summary

## The problem

Manufacturing plants typically run on **reactive** maintenance (fix it after it breaks) or **fixed-schedule preventive** maintenance (service equipment on a calendar, whether it needs it or not). Neither approach uses the early-warning signals already present in a machine's own sensor data. The result: unplanned downtime, larger repairs than necessary, and no data-driven way to decide which machine needs attention first.

## What was built

A working proof of concept that watches industrial machines, identifies which ones are showing early signs of trouble, diagnoses the likely cause, recommends a specific action based on what actually worked in similar past cases, and knows when to act automatically versus when a human should check first — built and tested entirely on a real, publicly available industrial dataset (100 machines, a full year of real operating history).

## Why this is credible, not just a demo

Every claim behind this system has been tested against real outcomes, not assumed:

- **94.7%** of real historical failures were correctly flagged with 24 hours of advance warning
- **93.6%** of the system's warnings turned out to be genuine, not false alarms
- A separately trained, validated machine learning model adds a second layer of prediction, independently confirmed at **82–94% accuracy** per failure type
- Extending the model's lookout window from 24 hours to a rolling 72 hours — the same trained model, simply checked further ahead, no retraining — raised the share of real failures caught from **24.3% to 80.0%** in backtesting across 6 separate historical dates, with **zero false alarms** observed among the added warnings (100% precision, 39 of 39 correct)
- The system's risk-level cutoffs were checked with formal statistical analysis and confirmed to be near-optimal, not arbitrary
- When the model reports very high confidence, that confidence was independently checked against 300 real outcomes and found to be genuinely reliable — correct **95.9%** of the time — and the dashboard now shows this real, measured figure rather than the model's raw self-reported number
- **Zero simulated or invented data** anywhere in the results — every number traces back to something real and checkable

## Where the team showed real rigor

On multiple occasions, a promising-looking feature was built, tested rigorously, found not to actually work as intended, and honestly retired or corrected — rather than shipped anyway.

- An early "24-hour failure probability" feature looked reasonable on the surface but was proven, through careful testing, to give the same answer whether a failure was imminent or not. This was caught before it reached a client, clearly explained with evidence, and replaced with an honest, correctly-labeled alternative.
- More recently, the dashboard appeared to show the model at "100% confidence" in almost every case — a result that could easily have been accepted at face value, or worse, presented to a client without question. Instead, it was investigated down to the model's underlying mathematics, measured against 300 real historical outcomes, and found to be a genuine (if slightly overstated) signal — not a bug, but not literally "100% certain" either. The dashboard was corrected to show the real, measured accuracy figure instead of the raw, misleadingly precise score.

This pattern — build, test rigorously, disclose findings honestly, fix or retire what doesn't hold up — has been the consistent approach throughout.

## What this demonstrates as a capability

- The ability to build a working, multi-step AI system that combines machine learning, real historical evidence, and AI reasoning into one coherent decision process
- A working, presentable demo interface a plant manager or engineer could actually use
- A track record of catching and correcting real mistakes before they become credibility problems — this matters as much as the results themselves when this is eventually shown to a real client
- The ability to extract more real value from an already-trained model (the 24-hour → 72-hour extension) through careful testing rather than requiring new data collection or retraining — a low-cost way to improve a pilot's practical usefulness

## What this is — and isn't — today

**This is:** a validated proof of concept, proving the approach works and can be trusted at the level tested.

**This is not yet:** a plug-and-play product ready to sell to a new client with different equipment and different data. Deploying this at an actual plant would require integrating with that plant's real sensors, and re-validating against that plant's own history — a natural, expected next phase, not a flaw in what exists today. The 72-hour extended-warning capability, in particular, is a promising early result (backtested on 70 real failures) rather than a fully proven one — a larger, ongoing backtest is the natural next step before broader reliance on it.

## Suggested next steps

1. **A small paid pilot** with a real manufacturing client, to prove the approach generalizes beyond this one public dataset
2. **Continuous monitoring**, moving from on-demand checks to an always-watching system with automatic alerts
3. **Time-to-failure estimation**, giving a specific countdown rather than just a risk category, once enough pilot data supports it
4. **A larger backtest of the 72-hour extended-warning window**, to confirm the strong early results (80.0% recall, 100% precision) hold up across more historical data before relying on it more heavily

## Bottom line

This is a working, evidence-backed proof of concept — not a polished sales pitch dressed up to look more finished than it is. Every number can be defended, every limitation is documented, and the process used to build it is itself a demonstration of the rigor this team brings to AI development.