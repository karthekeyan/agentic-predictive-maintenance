"""
FastAPI backend - wraps the existing agent pipeline behind web endpoints
for the React dashboard. No changes to the underlying agents/tools.
"""

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
import pandas as pd
import sys

sys.path.append('src')
sys.path.append('src/agents')
sys.path.append('src/tools')

from pipeline import build_pipeline
from predict_failure_risk import compute_health_scores, predict_failure_risk
from get_failure_rate_stats import calculate_mtbf_stats
from get_historical_failure_frequency import get_historical_failure_frequency
from predict_component_failure_24h import predict_component_failure_24h

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading data...")
telemetry = pd.read_csv("data/raw/PdM_telemetry.csv")
errors = pd.read_csv("data/raw/PdM_errors.csv")
maint = pd.read_csv("data/raw/PdM_maint.csv")
failures = pd.read_csv("data/raw/PdM_failures.csv")
machines = pd.read_csv("data/raw/PdM_machines.csv")

telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
errors['datetime'] = pd.to_datetime(errors['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])
maint['datetime'] = pd.to_datetime(maint['datetime'])

telemetry_scored = compute_health_scores(telemetry)
mtbf_table = calculate_mtbf_stats(failures, machines)
pipeline = build_pipeline()
print("Ready.")


@app.get("/machines-at-risk")
def machines_at_risk(date: str):
    as_of = pd.Timestamp(date)
    results = []

    for mid in machines['machineID'].unique():
        result = predict_failure_risk(int(mid), telemetry_scored, as_of=as_of)
        if result['health_score'] is not None:
            model = machines[machines['machineID'] == mid]['model'].values[0]
            freq_result = get_historical_failure_frequency(int(mid), model, as_of, failures, mtbf_table)

            results.append({
                "machineId": int(mid),
                "healthScore": round(float(result['health_score']), 4),
                "riskLevel": result['risk_level'],
                "avgDaysBetweenFailures": freq_result['avg_days_between_failures'],
                "failureFrequencyLabel": freq_result['frequency_label'],
                "model": model,
            })

    # Option B: run the real classifier only on High/Medium risk machines,
    # to get genuine failure-probability ranking without the cost of
    # running it on all 100 machines every time.
    high_medium = [r for r in results if r['riskLevel'] in ('high', 'medium')]
    low = [r for r in results if r['riskLevel'] == 'low']

    for r in high_medium:
        classifier_result = predict_component_failure_24h(
            machine_id=r['machineId'],
            as_of=as_of,
            telemetry_df=telemetry,
            errors_df=errors,
            maint_df=maint,
            machines_df=machines,
        )
        if classifier_result.get('status') == 'ok':
            # "none" means no failure predicted - treat its probability as
            # the inverse (low failure likelihood), so it sorts correctly
            # alongside real component predictions
            if classifier_result['predicted_label'] == 'none':
                r['failureProbability'] = 1 - classifier_result['probability']
            else:
                r['failureProbability'] = classifier_result['probability']
            r['classifierPrediction'] = classifier_result['predicted_label']
        else:
            r['failureProbability'] = 0
            r['classifierPrediction'] = None

    # Sort the High/Medium group by real failure probability (highest first)
    high_medium.sort(key=lambda x: -x.get('failureProbability', 0))

    # Low-risk machines stay sorted by health score, appended after
    low.sort(key=lambda x: -x['healthScore'])

    final_results = high_medium + low

    # Clean up the internal 'model' field we added, not needed in the response
    for r in final_results:
        r.pop('model', None)

    return {"date": date, "machines": final_results}

@app.get("/diagnose/{machine_id}")
def diagnose_machine(machine_id: int, date: str):
    as_of = pd.Timestamp(date)
    model = machines[machines['machineID'] == machine_id]['model'].values[0]

    state = {
        'machine_id': machine_id, 'model': model, 'as_of': as_of,
        'telemetry_scored': telemetry_scored,
        'telemetry_df': telemetry, 'errors_df': errors,
        'failures_df': failures,
        'maint_df': maint, 'machines_df': machines, 'mtbf_table': mtbf_table,
    }
    result = pipeline.invoke(state)

    # Classifier now runs INSIDE the pipeline's reasoning_node (Option C -
    # its prediction is fed into the LLM's prompt as evidence). We still
    # surface its raw output here separately, for transparency/audit.
    classifier_result = result.get('classifier_result', {})

    return {
        "machineId": machine_id,
        "healthScore": result['health_result']['health_score'],
        "riskLevel": result['health_result']['risk_level'],
        "diagnosis": result['diagnosis_result']['likely_component'],
        "confidence": result['diagnosis_result']['confidence'],
        "reasoning": result['diagnosis_result']['reasoning'],
        "recommendation": result['recommendation_result']['recommendation'],
        "routing": result['routing_result']['routing_decision'],
        "classifierPrediction": classifier_result.get('predicted_label'),
        "classifierProbability": classifier_result.get('probability'),
        "classifierStatus": classifier_result.get('status'),
    }