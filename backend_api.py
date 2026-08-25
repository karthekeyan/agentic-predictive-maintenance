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
from get_failure_probability import get_failure_probability

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],
    allow_methods=["*"],
    allow_headers=["*"],
)

print("Loading data...")
telemetry = pd.read_csv("data/raw/PdM_telemetry.csv")
maint = pd.read_csv("data/raw/PdM_maint.csv")
failures = pd.read_csv("data/raw/PdM_failures.csv")
machines = pd.read_csv("data/raw/PdM_machines.csv")

telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
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
            prob_result = get_failure_probability(int(mid), model, as_of, failures, mtbf_table)

            results.append({
                "machineId": int(mid),
                "healthScore": round(float(result['health_score']), 4),
                "riskLevel": result['risk_level'],
                "failureProbability24h": prob_result['probability_24h'],
                "failureRiskLabel": prob_result['risk_label'],
            })

    risk_order = {"high": 0, "medium": 1, "low": 2}
    results.sort(key=lambda x: -x['failureProbability24h'])
    return {"date": date, "machines": results}


@app.get("/diagnose/{machine_id}")
def diagnose_machine(machine_id: int, date: str):
    as_of = pd.Timestamp(date)
    model = machines[machines['machineID'] == machine_id]['model'].values[0]

    state = {
        'machine_id': machine_id, 'model': model, 'as_of': as_of,
        'telemetry_scored': telemetry_scored, 'failures_df': failures,
        'maint_df': maint, 'machines_df': machines, 'mtbf_table': mtbf_table,
    }
    result = pipeline.invoke(state)

    return {
        "machineId": machine_id,
        "healthScore": result['health_result']['health_score'],
        "riskLevel": result['health_result']['risk_level'],
        "diagnosis": result['diagnosis_result']['likely_component'],
        "confidence": result['diagnosis_result']['confidence'],
        "reasoning": result['diagnosis_result']['reasoning'],
        "recommendation": result['recommendation_result']['recommendation'],
        "routing": result['routing_result']['routing_decision'],
    }