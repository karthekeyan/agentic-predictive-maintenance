"""
Agent: Case Retrieval Agent
Given a machine showing risk, retrieves similar real historical cases from the
Knowledge Library. Builds a query using the SAME level of detail (individual sensor
deviations) as the stored case descriptions, so semantic matching has a fair basis
for comparison. Validates sensor data before proceeding - returns 'insufficient_data'
status rather than reasoning over NaN values (see Day 4 known issue).
"""

import sys
import pandas as pd
sys.path.append('../tools')
from retrieve_similar_cases import retrieve_similar_cases


def find_similar_past_cases(machine_id: int, telemetry_scored, as_of, priority_result: dict, n_results: int = 5) -> dict:
    current_data = telemetry_scored[
        (telemetry_scored['machineID'] == machine_id) &
        (telemetry_scored['datetime'] <= as_of)
    ]

    # Guard: check for valid, non-missing sensor data before proceeding
    if len(current_data) == 0:
        return {
            'machine_id': machine_id,
            'status': 'insufficient_data',
            'reason': 'No telemetry data available at or before the requested time.',
            'similar_cases': []
        }

    current = current_data.iloc[-1]
    check_cols = ['vibration_dev', 'volt_dev', 'pressure_dev', 'rotate_dev', 'health_risk_score']

    if current[check_cols].isna().any():
        return {
            'machine_id': machine_id,
            'status': 'insufficient_data',
            'reason': 'Sensor deviation data contains missing values (likely early in telemetry history or a data gap).',
            'similar_cases': []
        }

    # Proceed as normal once data is confirmed valid
    vibration_dev = current['vibration_dev']
    volt_dev = current['volt_dev']
    pressure_dev = current['pressure_dev']
    rotate_dev = current['rotate_dev']
    health_score = current['health_risk_score']

    likely_component = priority_result.get('most_likely_component')

    query = (
        f"Machine showing vibration deviated {vibration_dev*100:.1f}% from normal, "
        f"voltage {volt_dev*100:.1f}%, pressure {pressure_dev*100:.1f}%, "
        f"and rotation {rotate_dev*100:.1f}%. Combined health risk score averaged {health_score:.3f}."
    )

    broad_matches = retrieve_similar_cases(query_description=query, component_filter=None, n_results=n_results)

    filtered_matches = []
    if likely_component:
        filtered_matches = retrieve_similar_cases(query_description=query, component_filter=likely_component, n_results=3)

    combined = {}
    for match in broad_matches + filtered_matches:
        cid = match['case_id']
        if cid not in combined or match['similarity_distance'] < combined[cid]['similarity_distance']:
            combined[cid] = match

    sorted_matches = sorted(combined.values(), key=lambda x: x['similarity_distance'])

    return {
        'machine_id': machine_id,
        'status': 'ok',
        'query_used': query,
        'current_deviations': {
            'vibration_dev': round(vibration_dev, 4),
            'volt_dev': round(volt_dev, 4),
            'pressure_dev': round(pressure_dev, 4),
            'rotate_dev': round(rotate_dev, 4),
        },
        'statistically_likely_component': likely_component,
        'similar_cases': sorted_matches
    }