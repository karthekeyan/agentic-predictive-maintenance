"""
Feature engineering for the classical 24-hour failure prediction model.

This mirrors the batch feature-engineering logic from the
Classical-Predictive-Maintenance notebook EXACTLY, but computes features
for ONE machine at ONE point in time, instead of the whole dataset at once.
No methodology changes - same windows, same aggregations, same logic.

Source notebook steps this corresponds to:
  - Step 12.5e: telemetry_3h (min/max per 3-hour bucket)
  - Step 13.3: errors_24h_features (rolling 24h sum of one-hot error counts)
  - Step 14.3: days_since_features (days since last maintenance, per component)
  - Step 15.1: machine_features (age + one-hot model)
  - Step 16.1: master_df (merge of all the above)
"""

import pandas as pd

TELEMETRY_FIELDS = ['volt', 'rotate', 'pressure', 'vibration']
COMPONENTS = ['comp1', 'comp2', 'comp3', 'comp4']


def _get_3h_bucket_start(as_of: pd.Timestamp) -> pd.Timestamp:
    """
    Finds the start of the 3-hour bucket that `as_of` falls into, matching
    pandas' pd.Grouper(freq='3h') bucketing behavior (buckets start at
    00:00, 03:00, 06:00, ... each day).
    """
    floored = as_of.floor('3h')
    return floored


def build_telemetry_features(machine_id: int, as_of: pd.Timestamp, telemetry_df: pd.DataFrame) -> dict:
    """
    Step 12.5e equivalent: min/max of volt/rotate/pressure/vibration
    within the 3-hour bucket that `as_of` falls into.
    """
    bucket_start = _get_3h_bucket_start(as_of)
    bucket_end = bucket_start + pd.Timedelta(hours=3)

    window = telemetry_df[
        (telemetry_df['machineID'] == machine_id) &
        (telemetry_df['datetime'] >= bucket_start) &
        (telemetry_df['datetime'] < bucket_end)
    ]

    features = {}
    for field in TELEMETRY_FIELDS:
        if len(window) > 0:
            features[f'{field}_min_3h'] = window[field].min()
            features[f'{field}_max_3h'] = window[field].max()
        else:
            # No readings in this bucket - fall back to the most recent
            # available bucket before as_of, same spirit as the training
            # data (every machine has a reading in nearly every bucket in
            # practice; this is a safety net, not expected to trigger often)
            fallback = telemetry_df[
                (telemetry_df['machineID'] == machine_id) &
                (telemetry_df['datetime'] < bucket_end)
            ].sort_values('datetime').tail(3)
            if len(fallback) > 0:
                features[f'{field}_min_3h'] = fallback[field].min()
                features[f'{field}_max_3h'] = fallback[field].max()
            else:
                features[f'{field}_min_3h'] = None
                features[f'{field}_max_3h'] = None

    return features


def build_error_features(machine_id: int, as_of: pd.Timestamp, errors_df: pd.DataFrame) -> dict:
    """
    Step 13.3 equivalent: rolling 24-hour SUM of each one-hot error type,
    counted over the 24 hours ending at the bucket containing `as_of`.
    """
    bucket_start = _get_3h_bucket_start(as_of)
    bucket_end = bucket_start + pd.Timedelta(hours=3)
    window_start = bucket_end - pd.Timedelta(hours=24)

    machine_errors = errors_df[
        (errors_df['machineID'] == machine_id) &
        (errors_df['datetime'] >= window_start) &
        (errors_df['datetime'] < bucket_end)
    ]

    # Same one-hot approach as the notebook (Step 13.1) - build columns
    # for every errorID seen in the full dataset, not just this window,
    # so the output always has the same columns regardless of what
    # happened to occur in this specific 24h window.
    all_error_ids = sorted(errors_df['errorID'].unique())
    features = {}
    for eid in all_error_ids:
        col_name = f'errorID_{eid}_24h'
        features[col_name] = int((machine_errors['errorID'] == eid).sum())

    return features


def build_maintenance_features(machine_id: int, as_of: pd.Timestamp, maint_df: pd.DataFrame) -> dict:
    """
    Step 14.3 equivalent: days since the most recent replacement of each
    component (comp1-comp4), at or before `as_of`. Uses the same
    backward-looking logic as merge_asof(direction='backward').
    """
    features = {}
    machine_maint = maint_df[
        (maint_df['machineID'] == machine_id) &
        (maint_df['datetime'] <= as_of)
    ]

    for comp in COMPONENTS:
        comp_records = machine_maint[machine_maint['comp'] == comp]
        if len(comp_records) > 0:
            last_replaced = comp_records['datetime'].max()
            days_since = (as_of - last_replaced).total_seconds() / 86400
            features[f'days_since_{comp}'] = days_since
        else:
            # No prior replacement record for this component - matches
            # the NaN behavior noted in the notebook (Step 14.4) for
            # machines with no replacement before the dataset's start
            features[f'days_since_{comp}'] = None

    return features


def build_machine_features(machine_id: int, machines_df: pd.DataFrame, model_columns: list) -> dict:
    """
    Step 15.1 equivalent: age + one-hot encoded model, using the SAME
    set of model dummy columns the model was trained on (passed in via
    model_columns, read from feature_columns.json at inference time -
    this guarantees consistent one-hot columns even if this single
    machine's model type isn't otherwise inferable from context).
    """
    machine_row = machines_df[machines_df['machineID'] == machine_id].iloc[0]

    features = {'age': machine_row['age']}
    for col in model_columns:
        if col.startswith('model_'):
            model_value = col.replace('model_', '')
            features[col] = 1 if machine_row['model'] == model_value else 0

    return features


def build_features_for_prediction(
    machine_id: int,
    as_of: pd.Timestamp,
    telemetry_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    maint_df: pd.DataFrame,
    machines_df: pd.DataFrame,
    feature_columns: list,
) -> pd.DataFrame:
    """
    Builds a single-row feature vector for one machine at one point in
    time, matching the training-time feature set exactly.

    Args:
        machine_id: which machine to build features for
        as_of: the timestamp to build features as of (only data <= this
               timestamp is used - no future leakage, same discipline as
               the notebook's merge_asof(direction='backward') calls)
        telemetry_df, errors_df, maint_df, machines_df: the 4 real data
               tables (datetime columns must already be pd.Timestamp)
        feature_columns: the exact ordered list of columns the model was
               trained on (from feature_columns.json) - used to guarantee
               the output row has the right columns, in the right order

    Returns:
        A single-row DataFrame ready to pass into model.predict() /
        model.predict_proba(), with columns matching feature_columns.
    """
    telem_feats = build_telemetry_features(machine_id, as_of, telemetry_df)
    error_feats = build_error_features(machine_id, as_of, errors_df)
    maint_feats = build_maintenance_features(machine_id, as_of, maint_df)
    machine_feats = build_machine_features(machine_id, machines_df, feature_columns)

    all_features = {**telem_feats, **error_feats, **maint_feats, **machine_feats}

    row = pd.DataFrame([all_features])

    # Reindex to match training column order exactly - fills any missing
    # column with NaN rather than silently dropping/misaligning columns
    row = row.reindex(columns=feature_columns)

    return row
