"""
Tool: get_historical_maintenance_action
Looks up real historical maintenance actions for a given machine model + component,
based on actual maintenance records that followed real failures.
"""

import pandas as pd


def get_historical_maintenance_action(component: str, model: str, failures_df: pd.DataFrame,
                                        maint_df: pd.DataFrame, machines_df: pd.DataFrame) -> dict:
    """
    Given a likely failing component and machine model, returns real historical
    evidence of how this type of failure was resolved in the past.
    """
    # Find all real failures of this component, on this model
    model_machines = machines_df[machines_df['model'] == model]['machineID'].tolist()
    matching_failures = failures_df[
        (failures_df['failure'] == component) &
        (failures_df['machineID'].isin(model_machines))
    ]

    if len(matching_failures) == 0:
        return {
            'component': component,
            'model': model,
            'num_historical_cases': 0,
            'recommendation': f"No historical record of {component} failures on {model} to base a recommendation on.",
            'resolution_rate': None
        }

    # Check how many of these failures had a same-day maintenance record (confirms real resolution)
    resolved = matching_failures.merge(
        maint_df, left_on=['machineID', 'datetime', 'failure'],
        right_on=['machineID', 'datetime', 'comp'],
        how='left', indicator=True
    )
    resolved_count = (resolved['_merge'] == 'both').sum()
    total_count = len(matching_failures)
    resolution_rate = round(resolved_count / total_count * 100, 1)

    recommendation = (
        f"Service/replace {component}. In {resolved_count} of {total_count} historical "
        f"{component} failures on {model} machines ({resolution_rate}%), maintenance was "
        f"performed on the same component on the same day, confirming this as the standard resolution."
    )

    return {
        'component': component,
        'model': model,
        'num_historical_cases': total_count,
        'recommendation': recommendation,
        'resolution_rate': resolution_rate
    }