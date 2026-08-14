"""
Agent: Prioritization Agent
Ranks machine urgency by combining the Telemetry Health Agent's live health score
with empirical failure-rate statistics (which components are historically most at risk
for this machine's model).
"""

import sys
sys.path.append('../tools')
from get_failure_rate_stats import get_failure_rate_stats


def prioritize_machine(machine_id: int, model: str, health_result: dict, mtbf_table) -> dict:
    """
    Combines a machine's current health score with its model's known
    failure-rate profile to produce a single urgency score.
    """
    failure_profile = get_failure_rate_stats(model, mtbf_table)

    # Simple, transparent urgency formula:
    # higher health_score + lower avg_days_between_failures (for this model) = more urgent
    health_score = health_result.get('health_score') or 0
    highest_risk_days = failure_profile.get('highest_risk_avg_days')

    if highest_risk_days:
        urgency_score = health_score * (30 / highest_risk_days)  # normalize against a 30-day reference
    else:
        urgency_score = health_score

    return {
        'machine_id': machine_id,
        'model': model,
        'health_score': health_score,
        'risk_level': health_result.get('risk_level'),
        'most_likely_component': failure_profile.get('highest_risk_component'),
        'urgency_score': round(urgency_score, 4)
    }