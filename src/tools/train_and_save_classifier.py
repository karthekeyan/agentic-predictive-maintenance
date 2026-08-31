"""
Trains the classical 24-hour failure prediction model and saves it to disk,
so it can be loaded instantly by the tool instead of being retrained every time.

This reproduces the notebook's Steps 12-19 EXACTLY - same feature
engineering, same time-aware split (train < Nov 1 2015, test >= Nov 1 2015),
same model. The only difference from the notebook is that this saves the
trained model + feature column list to files at the end.

Run this ONCE from your project root:
    python train_and_save_classifier.py

Produces:
    models/classical_failure_model.pkl   - the trained XGBoost model
    models/feature_columns.json          - exact feature column order
    models/label_encoder.pkl             - encodes/decodes the failure labels
"""

import pandas as pd
import numpy as np
import json
import joblib
import os

from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

DATA_PATH = "data/raw/"
MODEL_DIR = "models/"

os.makedirs(MODEL_DIR, exist_ok=True)

print("Loading data...")
telemetry = pd.read_csv(DATA_PATH + "PdM_telemetry.csv")
errors = pd.read_csv(DATA_PATH + "PdM_errors.csv")
maint = pd.read_csv(DATA_PATH + "PdM_maint.csv")
machines = pd.read_csv(DATA_PATH + "PdM_machines.csv")
failures = pd.read_csv(DATA_PATH + "PdM_failures.csv")

telemetry['datetime'] = pd.to_datetime(telemetry['datetime'])
errors['datetime'] = pd.to_datetime(errors['datetime'])
maint['datetime'] = pd.to_datetime(maint['datetime'])
failures['datetime'] = pd.to_datetime(failures['datetime'])

# ---------------------------------------------------------------------------
# Step 12.5e equivalent: telemetry_3h - min/max per 3-hour bucket
# ---------------------------------------------------------------------------
print("Building telemetry features (3-hour buckets)...")
fields = ['volt', 'rotate', 'pressure', 'vibration']
telemetry_clean = telemetry[['datetime', 'machineID'] + fields].copy()

telemetry_3h = (
    telemetry_clean
    .groupby(['machineID', pd.Grouper(key='datetime', freq='3h')])[fields]
    .agg(['min', 'max'])
)
telemetry_3h.columns = [f'{field}_{stat}_3h' for field, stat in telemetry_3h.columns]
telemetry_3h = telemetry_3h.reset_index()

# ---------------------------------------------------------------------------
# Step 13.1-13.3 equivalent: errors_24h_features - rolling 24h sum
# ---------------------------------------------------------------------------
print("Building error features (rolling 24-hour counts)...")
errors_ohe = pd.get_dummies(errors[['datetime', 'machineID', 'errorID']], columns=['errorID'])
error_cols = [c for c in errors_ohe.columns if c.startswith('errorID_')]

errors_3h = (
    errors_ohe
    .groupby(['machineID', pd.Grouper(key='datetime', freq='3h')])[error_cols]
    .sum()
    .reset_index()
)

full_timeline = telemetry_3h[['machineID', 'datetime']].copy()
errors_full = full_timeline.merge(errors_3h, on=['machineID', 'datetime'], how='left')
errors_full[error_cols] = errors_full[error_cols].fillna(0)
errors_full = errors_full.sort_values(['machineID', 'datetime']).reset_index(drop=True)

errors_24h_features = (
    errors_full
    .groupby('machineID')[error_cols]
    .rolling(window=8, min_periods=1)
    .sum()
    .reset_index(level=0, drop=True)
)
errors_24h_features.columns = [f'{c}_24h' for c in errors_24h_features.columns]
errors_24h_features = pd.concat([errors_full[['machineID', 'datetime']], errors_24h_features], axis=1)

# ---------------------------------------------------------------------------
# Step 14.1-14.3 equivalent: days_since_features
# ---------------------------------------------------------------------------
print("Building maintenance features (days since last replacement)...")
maint_sorted = maint.sort_values(['machineID', 'datetime']).reset_index(drop=True)
components = ['comp1', 'comp2', 'comp3', 'comp4']

maint_by_comp = {}
for comp in components:
    comp_df = maint_sorted[maint_sorted['comp'] == comp][['machineID', 'datetime']].rename(
        columns={'datetime': f'{comp}_replaced_at'}
    )
    maint_by_comp[comp] = comp_df

full_timeline_sorted = telemetry_3h[['machineID', 'datetime']].sort_values(['machineID', 'datetime']).reset_index(drop=True)
days_since_features = full_timeline_sorted.copy()

for comp in components:
    comp_df = maint_by_comp[comp].sort_values(['machineID', f'{comp}_replaced_at']).reset_index(drop=True)
    merged = pd.merge_asof(
        full_timeline_sorted.sort_values('datetime'),
        comp_df.sort_values(f'{comp}_replaced_at'),
        left_on='datetime',
        right_on=f'{comp}_replaced_at',
        by='machineID',
        direction='backward'
    )
    merged[f'days_since_{comp}'] = (merged['datetime'] - merged[f'{comp}_replaced_at']).dt.total_seconds() / 86400
    days_since_features = days_since_features.merge(
        merged[['machineID', 'datetime', f'days_since_{comp}']],
        on=['machineID', 'datetime'],
        how='left'
    )

# ---------------------------------------------------------------------------
# Step 15.1 equivalent: machine_features
# ---------------------------------------------------------------------------
print("Building machine features (age, model)...")
machines_ohe_final = pd.get_dummies(machines, columns=['model'])
machine_features = full_timeline_sorted.merge(machines_ohe_final, on='machineID', how='left')

# ---------------------------------------------------------------------------
# Step 16.1 equivalent: master_df
# ---------------------------------------------------------------------------
print("Merging all feature tables...")
master_df = telemetry_3h.merge(errors_24h_features, on=['machineID', 'datetime'], how='left')
master_df = master_df.merge(days_since_features, on=['machineID', 'datetime'], how='left')
master_df = master_df.merge(machine_features, on=['machineID', 'datetime'], how='left')

# ---------------------------------------------------------------------------
# Step 17.1-17.4 equivalent: labeled_df - label construction
# ---------------------------------------------------------------------------
print("Building labels (failure within 24 hours)...")
failures_sorted = failures.sort_values('datetime').reset_index(drop=True)
master_df_sorted = master_df.sort_values('datetime').reset_index(drop=True)

failures_renamed = failures_sorted[['machineID', 'datetime', 'failure']].rename(
    columns={'datetime': 'next_failure_time', 'failure': 'next_failure_type'}
)

labeled_df = pd.merge_asof(
    master_df_sorted,
    failures_renamed,
    left_on='datetime',
    right_on='next_failure_time',
    by='machineID',
    direction='forward'
)

labeled_df['hours_to_failure'] = (
    labeled_df['next_failure_time'] - labeled_df['datetime']
).dt.total_seconds() / 3600

labeled_df['label'] = np.where(
    labeled_df['hours_to_failure'] <= 24,
    labeled_df['next_failure_type'],
    'none'
)

print("\nLabel distribution:")
print(labeled_df['label'].value_counts())

# ---------------------------------------------------------------------------
# Step 18.1 equivalent: time-aware train/test split
# ---------------------------------------------------------------------------
train_mask = labeled_df['datetime'] < '2015-11-01'
test_mask = labeled_df['datetime'] >= '2015-11-01'

train_df = labeled_df[train_mask].copy()
test_df = labeled_df[test_mask].copy()

# ---------------------------------------------------------------------------
# Step 19.1-19.2 equivalent: train XGBoost, save everything
# ---------------------------------------------------------------------------
print("\nTraining XGBoost model...")
drop_cols = ['machineID', 'datetime', 'next_failure_time', 'next_failure_type', 'hours_to_failure', 'label']

X_train = train_df.drop(columns=drop_cols)
y_train = train_df['label']
X_test = test_df.drop(columns=drop_cols)
y_test = test_df['label']

le = LabelEncoder()
y_train_enc = le.fit_transform(y_train)
y_test_enc = le.transform(y_test)

xgb_model = XGBClassifier(random_state=42, eval_metric='mlogloss')
xgb_model.fit(X_train, y_train_enc)

from sklearn.metrics import classification_report
y_pred = xgb_model.predict(X_test)
all_labels = np.arange(len(le.classes_))
print("\nTest set performance (should match the notebook's reported results):")
print(classification_report(y_test_enc, y_pred, labels=all_labels, target_names=le.classes_, zero_division=0))

# ---------------------------------------------------------------------------
# Save the model, feature columns, and label encoder
# ---------------------------------------------------------------------------
print("\nSaving model artifacts...")
joblib.dump(xgb_model, MODEL_DIR + "classical_failure_model.pkl")
joblib.dump(le, MODEL_DIR + "label_encoder.pkl")

with open(MODEL_DIR + "feature_columns.json", 'w') as f:
    json.dump(list(X_train.columns), f)

print(f"\nDone. Saved to {MODEL_DIR}:")
print("  - classical_failure_model.pkl")
print("  - label_encoder.pkl")
print("  - feature_columns.json")
