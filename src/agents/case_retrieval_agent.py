"""
Agent: Case Retrieval Agent
Given a machine showing risk, retrieves similar real historical cases from the
Knowledge Library. Builds a query using the SAME level of detail (individual sensor
deviations) as the stored case descriptions, so semantic matching has a fair basis
for comparison — matching a single combined score against detailed cases was found
to miss genuinely similar cases (e.g., comp4 cases never surfaced despite being the
correct answer for a real test case).
"""

import sys
sys.path.append('../tools')
from retrieve_similar_cases import retrieve_similar_cases


def find_similar_past_cases(machine_id: int, telemetry_scored, as_of, priority_result: dict, n_results: int = 5) -> dict:
    """
    Retrieves similar real past cases by building a query description that mirrors
    the stored case format — individual sensor deviations, not just a combined score.
    """
    current = telemetry_scored[
        (telemetry_scored['machineID'] == machine_id) &
        (telemetry_scored['datetime'] <= as_of)
    ].iloc[-1]

    vibration_dev = current['vibration_dev']
    volt_dev = current['volt_dev']
    pressure_dev = current['pressure_dev']
    rotate_dev = current['rotate_dev']
    health_score = current['health_risk_score']

    likely_component = priority_result.get('most_likely_component')

    # Same structure/detail level as the stored case descriptions
    query = (
        f"Machine showing vibration deviated {vibration_dev*100:.1f}% from normal, "
        f"voltage {volt_dev*100:.1f}%, pressure {pressure_dev*100:.1f}%, "
        f"and rotation {rotate_dev*100:.1f}%. Combined health risk score averaged {health_score:.3f}."
    )

    broad_matches = retrieve_similar_cases(
        query_description=query,
        component_filter=None,
        n_results=n_results
    )

    filtered_matches = []
    if likely_component:
        filtered_matches = retrieve_similar_cases(
            query_description=query,
            component_filter=likely_component,
            n_results=3
        )

    combined = {}
    for match in broad_matches + filtered_matches:
        cid = match['case_id']
        if cid not in combined or match['similarity_distance'] < combined[cid]['similarity_distance']:
            combined[cid] = match

    sorted_matches = sorted(combined.values(), key=lambda x: x['similarity_distance'])

    return {
        'machine_id': machine_id,
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