"""
Tool: predict_component_failure_24h
Predicts whether a machine will experience a component failure within the
next 24 hours, and which component, using a trained XGBoost classifier.

Model source: Classical-Predictive-Maintenance repo (karthekeyan).
Validated with a time-aware train/test split (train: Jan-Oct 2015,
test: Nov 2015-Jan 2016 - no shuffling, no future leakage) using 22
engineered features (telemetry min/max, rolling error counts, days since
maintenance, machine age/model). Reported test-set recall: 0.82-0.94
per component (comp1-comp4).

This tool loads the pre-trained model (see train_and_save_classifier.py)
rather than retraining on every call.
"""

import joblib
import json
import os
import pandas as pd

from build_features_for_prediction import build_features_for_prediction

MODEL_DIR = os.path.join(os.path.dirname(__file__), "..", "..", "models") + "/"

_model = None
_label_encoder = None
_feature_columns = None


def _load_artifacts():
    """Loads the trained model, label encoder, and feature column list once,
    caching them for subsequent calls (avoids reloading from disk every time)."""
    global _model, _label_encoder, _feature_columns

    if _model is None:
        _model = joblib.load(MODEL_DIR + "classical_failure_model.pkl")
        _label_encoder = joblib.load(MODEL_DIR + "label_encoder.pkl")
        with open(MODEL_DIR + "feature_columns.json") as f:
            _feature_columns = json.load(f)

    return _model, _label_encoder, _feature_columns


def predict_component_failure_24h(
    machine_id: int,
    as_of: pd.Timestamp,
    telemetry_df: pd.DataFrame,
    errors_df: pd.DataFrame,
    maint_df: pd.DataFrame,
    machines_df: pd.DataFrame,
) -> dict:
    """
    The actual tool function an agent calls.

    Args:
        machine_id: which machine to check
        as_of: the point in time to predict from (only data at or before
               this timestamp is used - matches the notebook's no-leakage
               discipline)
        telemetry_df, errors_df, maint_df, machines_df: the 4 real data
               tables (datetime columns must already be pd.Timestamp)

    Returns:
        dict with:
            machine_id
            predicted_label: 'none', 'comp1', 'comp2', 'comp3', or 'comp4'
            probability: the model's confidence in predicted_label
            all_probabilities: dict of every label -> its probability,
                                for full transparency
    """
    model, label_encoder, feature_columns = _load_artifacts()

    features = build_features_for_prediction(
        machine_id, as_of, telemetry_df, errors_df, maint_df, machines_df, feature_columns
    )

    # If any required feature is missing (e.g., no telemetry data yet for
    # this machine at this early a date), don't guess - same discipline as
    # the NaN validation guard already in the agentic pipeline
    if features.isna().any(axis=1).iloc[0]:
        missing = features.columns[features.isna().iloc[0]].tolist()
        return {
            'machine_id': machine_id,
            'status': 'insufficient_data',
            'reason': f'Missing feature values: {missing}',
            'predicted_label': None,
            'probability': None,
            'all_probabilities': {},
        }

    probabilities = model.predict_proba(features)[0]
    predicted_idx = probabilities.argmax()
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]

    all_probs = {
        label: float(prob)
        for label, prob in zip(label_encoder.classes_, probabilities)
    }

    return {
        'machine_id': machine_id,
        'status': 'ok',
        'predicted_label': predicted_label,
        'probability': float(probabilities[predicted_idx]),
        'all_probabilities': all_probs,
    }
