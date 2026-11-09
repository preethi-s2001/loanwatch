"""
Unit tests for Phase 4 Model Training, Artifact Serialization, and Calibration Integrity.
"""

import json
import joblib
import numpy as np
import pandas as pd
import pytest

from src.config import (
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MODEL_FEATURE_COLUMNS,
    MODEL_PATH,
    SAMPLE_DATA_PATH,
)
from src.features import compute_point_in_time_features


def test_model_artifacts_exist():
    """Verify serialized model, feature columns, and metrics.json are present."""
    assert MODEL_PATH.exists(), "Trained model artifact missing."
    assert FEATURE_COLUMNS_PATH.exists(), "Feature columns JSON missing."
    assert METRICS_PATH.exists(), "Metrics JSON missing."


def test_model_loading_and_inference():
    """Verify model can be deserialized and predicts valid probabilities."""
    model = joblib.load(MODEL_PATH)

    with open(FEATURE_COLUMNS_PATH, "r") as f:
        feature_cols = json.load(f)

    assert feature_cols == MODEL_FEATURE_COLUMNS

    # Load test batch from clean sample
    df_raw = pd.read_csv(SAMPLE_DATA_PATH).head(100)
    df_feat = compute_point_in_time_features(df_raw)
    X = df_feat[feature_cols]

    probs = model.predict_proba(X)[:, 1]

    # Verify probability bounds
    assert len(probs) == 100
    assert (probs >= 0.0).all() and (probs <= 1.0).all()
    assert np.isnan(probs).sum() == 0


def test_metrics_integrity():
    """Verify metrics payload contains required models and splits."""
    with open(METRICS_PATH, "r") as f:
        metrics = json.load(f)

    assert "split_summary" in metrics
    assert "model_comparison" in metrics
    assert "calibration_analysis" in metrics
    assert "recommended_model" in metrics

    # Check that test split had >= 300 positives
    test_positives = metrics["split_summary"]["Test_OOT"]["positives"]
    assert test_positives >= 300, f"Expected >= 300 positives, got {test_positives}"

    # Check calibrated RF metrics
    rf_cal = metrics["model_comparison"]["Random_Forest_Calibrated"]
    assert rf_cal["pr_auc"] > 0.65
    assert rf_cal["brier_score"] < 0.08
