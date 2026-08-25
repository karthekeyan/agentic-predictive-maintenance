"""
Tool: get_failure_probability
Calculates the probability a machine fails within the next 24 hours, using
real gaps between THIS machine's own historical failure dates when there's
enough data (3+ failures), otherwise falling back to model-level MTBF stats.
"""

import pandas as pd

MIN_FAILURES_FOR_OWN_HISTORY = 3

HIGH_THRESHOLD = 0.10
MEDIUM_THRESHOLD = 0.03


def get_failure_probability(machine_id: int, model: str, as_of, failures_df: pd.DataFrame,
                              mtbf_table: pd.DataFrame) -> dict:
    machine_failures = failures_df[
        (failures_df['machineID'] == machine_id) &
        (failures_df['datetime'] <= as_of)
    ].sort_values('datetime')

    num_failures = len(machine_failures)

    if num_failures >= MIN_FAILURES_FOR_OWN_HISTORY:
        gaps = machine_failures['datetime'].diff().dropna()
        avg_days_between_failures = gaps.mean().total_seconds() / 86400
        method = 'own_history'
    else:
        model_stats = mtbf_table[mtbf_table['model'] == model]
        if len(model_stats) > 0:
            avg_days_between_failures = model_stats['avg_days_between_failures'].mean()
            method = 'model_fallback'
        else:
            avg_days_between_failures = 365
            method = 'no_data_fallback'

    probability_24h = 1 / avg_days_between_failures if avg_days_between_failures > 0 else 0

    if probability_24h > HIGH_THRESHOLD:
        risk_label = 'high'
    elif probability_24h > MEDIUM_THRESHOLD:
        risk_label = 'medium'
    else:
        risk_label = 'low'

    return {
        'machine_id': machine_id,
        'num_own_failures': num_failures,
        'method': method,
        'avg_days_between_failures': round(avg_days_between_failures, 1),
        'probability_24h': round(probability_24h, 4),
        'risk_label': risk_label,
    }