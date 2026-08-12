"""
Agent: Telemetry Health Agent
Decides which machines to check and interprets the prediction tool's output.
Calls the predict_failure_risk tool rather than containing any ML logic itself.
"""

import sys
sys.path.append('../tools')
from predict_failure_risk import predict_failure_risk


def check_machine_health(machine_id: int, telemetry_with_scores, as_of=None) -> dict:
    """
    The agent's main entry point. Calls the prediction tool, then adds
    a layer of interpretation on top of the raw result.
    """
    result = predict_failure_risk(machine_id, telemetry_with_scores, as_of=as_of)

    # Agent-level reasoning: decide what action this result warrants
    if result['risk_level'] == 'high':
        result['recommended_next_step'] = 'escalate_to_prioritization'
    elif result['risk_level'] == 'medium':
        result['recommended_next_step'] = 'monitor_closely'
    else:
        result['recommended_next_step'] = 'no_action_needed'

    return result