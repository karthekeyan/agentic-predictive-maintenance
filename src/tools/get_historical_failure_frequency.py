"""
Tool: get_historical_failure_frequency
Reports how often a machine has historically failed - using real gaps
between THIS machine's own historical failure dates when there's enough
data (3+ failures), otherwise falling back to model-level MTBF stats.

NOTE: this was originally built and tested as a 24-hour failure probability
predictor. Backtesting on 25th Aug showed it has no real predictive power
for a specific 24h window (average calculated probability was nearly
identical before real failures vs. ordinary days: 8.02% vs 7.92%, and a
more sophisticated elapsed-time version also failed: 15.42% vs 16.39%).
Simplified to report only what's actually validated: historical failure
frequency, used as a secondary tie-breaker alongside the health score -
not a day-specific prediction. See README "Known limitations" for detail.
"""

import pandas as pd

MIN_FAILURES_FOR_OWN_HISTORY = 3

# Labels describing how failure-prone this machine has historically been -
# NOT a probability of failing on any specific day
FREQUENT_THRESHOLD_DAYS = 15   # fails more often than every 15 days -> "high" frequency
MODERATE_THRESHOLD_DAYS = 40   # fails every 15-40 days -> "medium" frequency
                                 # less often than every 40 days -> "low" frequency


def get_historical_failure_frequency(machine_id: int, model: str, as_of, failures_df: pd.DataFrame,
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

    if avg_days_between_failures <= FREQUENT_THRESHOLD_DAYS:
        frequency_label = 'high'
    elif avg_days_between_failures <= MODERATE_THRESHOLD_DAYS:
        frequency_label = 'medium'
    else:
        frequency_label = 'low'

    return {
        'machine_id': machine_id,
        'num_own_failures': num_failures,
        'method': method,
        'avg_days_between_failures': round(avg_days_between_failures, 1),
        'frequency_label': frequency_label,
    }