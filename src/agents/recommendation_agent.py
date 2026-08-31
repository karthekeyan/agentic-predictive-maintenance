"""
Agent: Recommendation Agent
Given a diagnosis (likely failing component) from the Reasoning Agent, retrieves
real historical evidence of how similar failures were resolved and produces a
grounded maintenance recommendation.
"""

import sys
sys.path.append('../tools')
from get_historical_maintenance_action import get_historical_maintenance_action


def recommend_action(machine_id: int, model: str, diagnosis_result: dict,
                       failures_df, maint_df, machines_df) -> dict:
    """
    Takes the Reasoning Agent's diagnosis and produces a maintenance
    recommendation grounded in real historical repair outcomes.
    """
    component = diagnosis_result.get('likely_component')

    if component is None:
        return {
            'machine_id': machine_id,
            'recommendation': 'No recommendation - diagnosis was inconclusive or skipped.',
            'based_on_cases': 0
        }

    if component.lower() == 'none':
        return {
            'machine_id': machine_id,
            'component': 'none',
            'diagnosis_confidence': diagnosis_result.get('confidence'),
            'recommendation': 'No maintenance action needed - no failure predicted for this machine.',
            'based_on_cases': 0,
            'historical_resolution_rate': None
        }

    action_result = get_historical_maintenance_action(component, model, failures_df, maint_df, machines_df)

    return {
        'machine_id': machine_id,
        'component': component,
        'diagnosis_confidence': diagnosis_result.get('confidence'),
        'recommendation': action_result['recommendation'],
        'based_on_cases': action_result['num_historical_cases'],
        'historical_resolution_rate': action_result['resolution_rate']
    }

