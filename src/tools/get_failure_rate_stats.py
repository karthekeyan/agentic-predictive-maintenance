"""
Tool: get_failure_rate_stats
Calculates empirical failure-rate statistics (MTBF-style) per machine model
and component, based on real historical failure records.
"""

import pandas as pd


def calculate_mtbf_stats(failures_df: pd.DataFrame, machines_df: pd.DataFrame) -> pd.DataFrame:
    """Fleet-wide average days between failures, per model + component."""
    merged = failures_df.merge(machines_df, on='machineID')
    merged = merged.sort_values(['model', 'failure', 'datetime'])

    mtbf_records = []
    for (model, component), group in merged.groupby(['model', 'failure']):
        dates = group['datetime'].sort_values()
        if len(dates) > 1:
            gaps = dates.diff().dropna()
            avg_gap_days = gaps.mean().total_seconds() / 86400
            mtbf_records.append({
                'model': model,
                'component': component,
                'num_failures_observed': len(dates),
                'avg_days_between_failures': round(avg_gap_days, 1)
            })

    return pd.DataFrame(mtbf_records)


def get_failure_rate_stats(model: str, mtbf_table: pd.DataFrame) -> dict:
    """
    The actual tool function an agent calls.
    Given a machine's model, returns its known failure-rate profile
    across all components, sorted by highest risk first.
    """
    model_stats = mtbf_table[mtbf_table['model'] == model].sort_values('avg_days_between_failures')

    if len(model_stats) == 0:
        return {'model': model, 'components': [], 'highest_risk_component': None}

    return {
        'model': model,
        'components': model_stats.to_dict('records'),
        'highest_risk_component': model_stats.iloc[0]['component'],
        'highest_risk_avg_days': model_stats.iloc[0]['avg_days_between_failures']
    }