"""
Agent: Case Retrieval Agent
Given a machine showing risk, retrieves similar real historical cases from the
Knowledge Library. Uses dominant_sensor as a structured filter (Day 6 fix) since
pure text-embedding search could not reliably distinguish numeric magnitude
(e.g., comp1's voltage-dominant signature was never surfacing correctly).
dominant_sensor matches are prioritized and never outranked by broader,
less-reliable text-distance-only matches (second Day 6 fix - merge-sort bug).
Validates sensor data before proceeding - returns 'insufficient_data' status
rather than reasoning over NaN values (Day 4/5 fix).
"""

import sys
import pandas as pd
sys.path.append('../tools')
from retrieve_similar_cases import retrieve_similar_cases, compute_dominant_sensor


def find_similar_past_cases(machine_id: int, telemetry_scored, as_of, priority_result: dict, n_results: int = 5) -> dict:
    current_data = telemetry_scored[
        (telemetry_scored['machineID'] == machine_id) &
        (telemetry_scored['datetime'] <= as_of)
    ]

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

    vibration_dev = current['vibration_dev']
    volt_dev = current['volt_dev']
    pressure_dev = current['pressure_dev']
    rotate_dev = current['rotate_dev']
    health_score = current['health_risk_score']

    likely_component = priority_result.get('most_likely_component')
    query_dominant_sensor = compute_dominant_sensor(vibration_dev, volt_dev, pressure_dev, rotate_dev)

    query = (
        f"Machine showing vibration deviated {vibration_dev*100:.1f}% from normal, "
        f"voltage {volt_dev*100:.1f}%, pressure {pressure_dev*100:.1f}%, "
        f"and rotation {rotate_dev*100:.1f}%. Combined health risk score averaged {health_score:.3f}."
    )

    # Primary search: filter by the query's own dominant sensor (the key fix)
    dominant_matches = retrieve_similar_cases(
        query_description=query, dominant_sensor_filter=query_dominant_sensor, n_results=n_results
    )

    if len(dominant_matches) >= n_results:
        # Enough good matches found - use them as-is, don't dilute with broader search
        sorted_matches = dominant_matches
    else:
        # Not enough dominant-sensor matches - fill remaining slots from broader searches,
        # but dominant matches always rank first, never outranked by raw text-distance
        broad_matches = retrieve_similar_cases(query_description=query, n_results=n_results)
        filtered_matches = []
        if likely_component:
            filtered_matches = retrieve_similar_cases(query_description=query, component_filter=likely_component, n_results=3)

        dominant_ids = {m['case_id'] for m in dominant_matches}
        combined = {m['case_id']: m for m in dominant_matches}
        for match in broad_matches + filtered_matches:
            cid = match['case_id']
            if cid not in combined:
                combined[cid] = match

        sorted_matches = sorted(
            combined.values(),
            key=lambda x: (x['case_id'] not in dominant_ids, x['similarity_distance'])
        )

    return {
        'machine_id': machine_id,
        'status': 'ok',
        'query_used': query,
        'query_dominant_sensor': query_dominant_sensor,
        'current_deviations': {
            'vibration_dev': round(vibration_dev, 4),
            'volt_dev': round(volt_dev, 4),
            'pressure_dev': round(pressure_dev, 4),
            'rotate_dev': round(rotate_dev, 4),
        },
        'statistically_likely_component': likely_component,
        'similar_cases': sorted_matches
    }