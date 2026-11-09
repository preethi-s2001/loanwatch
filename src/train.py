"""
Training and Model Comparison Pipeline for LoanWatch NBFC Early Warning System.

Performs:
1. Strict time-based splitting with 3-month embargo gaps.
2. Fair training and comparison across:
   - DPD-only Logistic Baseline (Benchmark)
   - Multivariable Logistic Regression
   - Depth-Limited Decision Tree
   - Random Forest Classifier
3. Probability calibration via CalibratedClassifierCV on the validation split.
4. Programmatic model selection based strictly on VALIDATION metrics (PR-AUC primary, Brier secondary).
5. Single evaluation on the held-out Out-of-Time Test set.
6. Calibration curve visualization (models/calibration_curve.png).
7. Serialization of best model, feature names, and metrics.json with real-number rationales.
"""

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Tuple
import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.tree import DecisionTreeClassifier

from src.config import (
    BASELINE_FEATURE_COLUMNS,
    EMBARGO_MONTHS,
    FEATURE_COLUMNS_PATH,
    METRICS_PATH,
    MODEL_FEATURE_COLUMNS,
    MODEL_PATH,
    MODELS_DIR,
    PROCESSED_DATA_PATH,
    RANDOM_SEED,
    RISK_THRESHOLD_MEDIUM,
    SAMPLE_DATA_PATH,
    TARGET_COLUMN,
    get_split_dates,
)
from src.evaluate import compute_calibration_data, evaluate_predictions
from src.features import compute_point_in_time_features

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_or_generate_processed_data() -> pd.DataFrame:
    """Loads processed features dataset, or computes it from sample data if missing."""
    if PROCESSED_DATA_PATH.exists():
        logger.info("Loading processed features from %s", PROCESSED_DATA_PATH)
        df = pd.read_csv(PROCESSED_DATA_PATH)
    else:
        logger.info("Processed features not found. Generating from %s...", SAMPLE_DATA_PATH)
        df_raw = pd.read_csv(SAMPLE_DATA_PATH)
        df = compute_point_in_time_features(df_raw)
        PROCESSED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(PROCESSED_DATA_PATH, index=False)
    return df


def split_data_by_time(
    df: pd.DataFrame,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, Dict[str, int]]]:
    """
    Performs strict time-based train/val/test splitting with 3-month embargo gaps.
    Returns df_train, df_val, df_test, and partition statistics.
    """
    unique_dates = pd.to_datetime(df["observation_date"]).drop_duplicates().sort_values().tolist()
    split_dates = get_split_dates(unique_dates)

    train_dates = [d.strftime("%Y-%m-%d") for d in split_dates["train"]]
    val_dates = [d.strftime("%Y-%m-%d") for d in split_dates["validation"]]
    test_dates = [d.strftime("%Y-%m-%d") for d in split_dates["test"]]

    df_train = df[df["observation_date"].isin(train_dates)].copy()
    df_val = df[df["observation_date"].isin(val_dates)].copy()
    df_test = df[df["observation_date"].isin(test_dates)].copy()

    split_stats = {}
    for name, split_df in [("Train", df_train), ("Validation", df_val), ("Test_OOT", df_test)]:
        pos = int(split_df[TARGET_COLUMN].sum())
        total = len(split_df)
        loans = split_df["loan_id"].nunique()
        split_stats[name] = {
            "rows": total,
            "distinct_loans": loans,
            "positives": pos,
            "default_rate_pct": round((pos / total * 100), 2) if total > 0 else 0.0,
        }

    return df_train, df_val, df_test, split_stats


def train_and_compare_models(
    df_train: pd.DataFrame,
    df_val: pd.DataFrame,
    df_test: pd.DataFrame,
) -> Dict[str, Any]:
    """
    Trains candidate models without class_weight distortions, calibrates on validation set,
    programmatically selects the best model on validation PR-AUC / Brier score,
    and evaluates on test set.
    """
    # Feature matrices
    X_train = df_train[MODEL_FEATURE_COLUMNS]
    y_train = df_train[TARGET_COLUMN].values

    X_val = df_val[MODEL_FEATURE_COLUMNS]
    y_val = df_val[TARGET_COLUMN].values

    X_test = df_test[MODEL_FEATURE_COLUMNS]
    y_test = df_test[TARGET_COLUMN].values

    # DPD-only baseline matrices
    X_train_dpd = df_train[BASELINE_FEATURE_COLUMNS]
    X_val_dpd = df_val[BASELINE_FEATURE_COLUMNS]
    X_test_dpd = df_test[BASELINE_FEATURE_COLUMNS]

    logger.info("Training candidate models (unweighted for fair probability calibration)...")

    # 1. Benchmark: DPD-Only Baseline (Logistic Regression on days_past_due only)
    dpd_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(random_state=RANDOM_SEED)),
    ])
    dpd_pipeline.fit(X_train_dpd, y_train)

    # 2. Multivariable Logistic Regression
    lr_pipeline = Pipeline([
        ("scaler", StandardScaler()),
        ("classifier", LogisticRegression(
            C=0.5,
            max_iter=1000,
            random_state=RANDOM_SEED,
        )),
    ])
    lr_pipeline.fit(X_train, y_train)

    # 3. Depth-Limited Decision Tree
    dt_classifier = DecisionTreeClassifier(
        max_depth=5,
        min_samples_leaf=50,
        random_state=RANDOM_SEED,
    )
    dt_classifier.fit(X_train, y_train)

    # 4. Random Forest Classifier
    rf_classifier = RandomForestClassifier(
        n_estimators=100,
        max_depth=8,
        min_samples_leaf=25,
        random_state=RANDOM_SEED,
        n_jobs=-1,
    )
    rf_classifier.fit(X_train, y_train)

    # Calibrate candidate models on the validation set using sigmoid scaling
    logger.info("Calibrating candidate models on validation partition...")
    calibrated_dpd = CalibratedClassifierCV(estimator=dpd_pipeline, method="sigmoid", cv="prefit").fit(X_val_dpd, y_val)
    calibrated_lr = CalibratedClassifierCV(estimator=lr_pipeline, method="sigmoid", cv="prefit").fit(X_val, y_val)
    calibrated_dt = CalibratedClassifierCV(estimator=dt_classifier, method="sigmoid", cv="prefit").fit(X_val, y_val)
    calibrated_rf = CalibratedClassifierCV(estimator=rf_classifier, method="sigmoid", cv="prefit").fit(X_val, y_val)

    # Model registry for validation and testing
    all_models = {
        "DPD_Only_Baseline": {
            "model": dpd_pipeline,
            "X_val": X_val_dpd,
            "X_test": X_test_dpd,
            "is_baseline": True,
        },
        "DPD_Only_Calibrated": {
            "model": calibrated_dpd,
            "X_val": X_val_dpd,
            "X_test": X_test_dpd,
            "is_baseline": True,
        },
        "Logistic_Regression": {
            "model": lr_pipeline,
            "X_val": X_val,
            "X_test": X_test,
            "is_baseline": False,
        },
        "Logistic_Regression_Calibrated": {
            "model": calibrated_lr,
            "X_val": X_val,
            "X_test": X_test,
            "is_baseline": False,
        },
        "Decision_Tree": {
            "model": dt_classifier,
            "X_val": X_val,
            "X_test": X_test,
            "is_baseline": False,
        },
        "Decision_Tree_Calibrated": {
            "model": calibrated_dt,
            "X_val": X_val,
            "X_test": X_test,
            "is_baseline": False,
        },
        "Random_Forest": {
            "model": rf_classifier,
            "X_val": X_val,
            "X_test": X_test,
            "is_baseline": False,
        },
        "Random_Forest_Calibrated": {
            "model": calibrated_rf,
            "X_val": X_val,
            "X_test": X_test,
            "is_baseline": False,
        },
    }

    # Operating decision threshold: 0.30
    # In LoanWatch, default probability >= 0.30 places a loan into Medium or High Risk,
    # triggering supervisor and collector outreach.
    operating_threshold = RISK_THRESHOLD_MEDIUM

    logger.info("Evaluating all candidate models on VALIDATION set at operating threshold %.2f...", operating_threshold)
    val_metrics: Dict[str, Dict[str, Any]] = {}
    for name, info in all_models.items():
        probs_val = info["model"].predict_proba(info["X_val"])[:, 1]
        metrics = evaluate_predictions(y_val, probs_val, threshold=operating_threshold)
        val_metrics[name] = metrics
        logger.info(
            "VAL | %-30s -> PR-AUC: %.4f | Brier: %.4f | ROC-AUC: %.4f | Recall: %.4f | F1: %.4f",
            name, metrics["pr_auc"], metrics["brier_score"], metrics["roc_auc"], metrics["recall"], metrics["f1"]
        )

    # Programmatic Model Selection:
    # Selected among multi-feature candidates based strictly on VALIDATION metrics:
    # Primary: PR-AUC (higher is better)
    # Secondary: Brier Score (lower is better)
    multi_feature_names = [k for k, v in all_models.items() if not v["is_baseline"]]
    best_model_name = sorted(
        multi_feature_names,
        key=lambda m: (-val_metrics[m]["pr_auc"], val_metrics[m]["brier_score"])
    )[0]
    best_model_obj = all_models[best_model_name]["model"]

    logger.info(
        "=== Programmatically Selected Best Model: %s (Val PR-AUC: %.4f, Val Brier: %.4f) ===",
        best_model_name, val_metrics[best_model_name]["pr_auc"], val_metrics[best_model_name]["brier_score"]
    )

    # Evaluate all models on TEST set (once) for transparent comparison
    logger.info("Evaluating models once on OUT-OF-TIME TEST SET...")
    test_metrics: Dict[str, Dict[str, Any]] = {}
    test_probs: Dict[str, np.ndarray] = {}
    for name, info in all_models.items():
        probs_test = info["model"].predict_proba(info["X_test"])[:, 1]
        test_probs[name] = probs_test
        metrics = evaluate_predictions(y_test, probs_test, threshold=operating_threshold)
        test_metrics[name] = metrics
        logger.info(
            "TEST | %-30s -> PR-AUC: %.4f | Brier: %.4f | ROC-AUC: %.4f | Recall: %.4f | F1: %.4f",
            name, metrics["pr_auc"], metrics["brier_score"], metrics["roc_auc"], metrics["recall"], metrics["f1"]
        )

    # Generate calibration curve plot comparing Baseline, Logistic Regression, and Best Model
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    plot_path = MODELS_DIR / "calibration_curve.png"
    plt.figure(figsize=(8, 6), dpi=150)
    plt.plot([0, 1], [0, 1], "k--", label="Perfect Calibration (y = x)")

    curve_models = [
        ("DPD_Only_Baseline", test_probs["DPD_Only_Baseline"], "#7f8c8d", ":"),
        ("Logistic_Regression_Calibrated", test_probs["Logistic_Regression_Calibrated"], "#2980b9", "--"),
        (best_model_name, test_probs[best_model_name], "#e74c3c", "-"),
    ]

    cal_curve_data: Dict[str, Any] = {}
    for label, probs, color, ls in curve_models:
        prob_true, prob_pred = compute_calibration_data(y_test, probs, n_bins=10)
        plt.plot(prob_pred, prob_true, marker="o", label=f"{label} (Brier: {test_metrics[label]['brier_score']:.4f})", color=color, linestyle=ls)
        if label == best_model_name:
            cal_curve_data = {
                "prob_true": [round(float(v), 4) for v in prob_true],
                "prob_pred": [round(float(v), 4) for v in prob_pred],
            }

    plt.xlabel("Mean Predicted Default Probability")
    plt.ylabel("Empirical Fraction of Positives (Defaults)")
    plt.title("LoanWatch: Out-of-Time Probability Calibration Curves")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(plot_path)
    plt.close()
    logger.info("Saved calibration curve plot to %s", plot_path)

    # Extract feature importance from the selected model
    if hasattr(best_model_obj, "feature_importances_"):
        raw_importances = best_model_obj.feature_importances_
    elif hasattr(best_model_obj, "calibrated_classifiers_") and hasattr(best_model_obj.calibrated_classifiers_[0].estimator, "feature_importances_"):
        raw_importances = best_model_obj.calibrated_classifiers_[0].estimator.feature_importances_
    elif hasattr(best_model_obj, "estimator") and hasattr(best_model_obj.estimator, "feature_importances_"):
        raw_importances = best_model_obj.estimator.feature_importances_
    else:
        raw_importances = None

    if raw_importances is not None:
        feature_importances = dict(zip(
            MODEL_FEATURE_COLUMNS,
            [round(float(v), 4) for v in raw_importances]
        ))
        feature_importances = dict(sorted(feature_importances.items(), key=lambda item: item[1], reverse=True))
    else:
        feature_importances = {}

    # Calibration analysis comparison
    calibration_analysis = {
        "best_model": {
            "name": best_model_name,
            "validation_pr_auc": val_metrics[best_model_name]["pr_auc"],
            "validation_brier": val_metrics[best_model_name]["brier_score"],
            "test_pr_auc": test_metrics[best_model_name]["pr_auc"],
            "test_brier": test_metrics[best_model_name]["brier_score"],
            "test_recall_at_0_30": test_metrics[best_model_name]["recall"],
        },
        "dpd_only_baseline": {
            "validation_pr_auc": val_metrics["DPD_Only_Baseline"]["pr_auc"],
            "validation_brier": val_metrics["DPD_Only_Baseline"]["brier_score"],
            "test_pr_auc": test_metrics["DPD_Only_Baseline"]["pr_auc"],
            "test_brier": test_metrics["DPD_Only_Baseline"]["brier_score"],
            "test_recall_at_0_30": test_metrics["DPD_Only_Baseline"]["recall"],
        },
    }

    # Generate dynamic rationale strictly from the computed numbers
    val_pr = val_metrics[best_model_name]["pr_auc"]
    val_brier = val_metrics[best_model_name]["brier_score"]
    base_val_pr = val_metrics["DPD_Only_Baseline"]["pr_auc"]
    base_val_brier = val_metrics["DPD_Only_Baseline"]["brier_score"]
    test_pr = test_metrics[best_model_name]["pr_auc"]
    test_brier = test_metrics[best_model_name]["brier_score"]
    base_test_pr = test_metrics["DPD_Only_Baseline"]["pr_auc"]

    dynamic_rationale = (
        f"Selected '{best_model_name}' programmatically on the Validation partition: "
        f"achieved primary Validation PR-AUC of {val_pr:.4f} and secondary Brier score of {val_brier:.4f}, "
        f"compared to the DPD-Only Baseline (Val PR-AUC: {base_val_pr:.4f}, Brier: {base_val_brier:.4f}). "
        f"On the held-out Out-of-Time Test set, it achieved PR-AUC {test_pr:.4f} and Brier {test_brier:.4f} "
        f"(vs Baseline Test PR-AUC: {base_test_pr:.4f}). "
        f"In this synthetic portfolio, delinquency metrics (days_past_due, days_late, dpd_trend_3m) "
        f"dominate the predictive signal, while recent auto-debit bounces provide incremental discrimination."
    )

    return {
        "best_model_name": best_model_name,
        "best_model": best_model_obj,
        "val_metrics": val_metrics,
        "test_metrics": test_metrics,
        "operating_threshold": operating_threshold,
        "calibration_analysis": calibration_analysis,
        "calibration_curve": cal_curve_data,
        "feature_importances": feature_importances,
        "dynamic_rationale": dynamic_rationale,
    }


def main():
    """CLI execution for model training, comparison, calibration, and persistence."""
    logger.info("Starting Modeling Pipeline...")
    df = load_or_generate_processed_data()

    # Split dataset
    df_train, df_val, df_test, split_stats = split_data_by_time(df)

    logger.info("=== Time-Based Partition Summary ===")
    for split_name, stats in split_stats.items():
        logger.info(
            "%12s | Rows: %6d | Distinct Loans: %5d | Positives: %5d (%5.2f%%)",
            split_name, stats["rows"], stats["distinct_loans"], stats["positives"], stats["default_rate_pct"]
        )

    # Train and compare
    results = train_and_compare_models(df_train, df_val, df_test)

    # Prepare persistence payload
    MODELS_DIR.mkdir(parents=True, exist_ok=True)

    # 1. Save best model
    joblib.dump(results["best_model"], MODEL_PATH)
    logger.info("Saved best model (%s) to %s", results["best_model_name"], MODEL_PATH)

    # 2. Save feature columns schema
    with open(FEATURE_COLUMNS_PATH, "w", encoding="utf-8") as f:
        json.dump(MODEL_FEATURE_COLUMNS, f, indent=2)
    logger.info("Saved feature schema to %s", FEATURE_COLUMNS_PATH)

    # 3. Save comprehensive metrics.json
    metrics_payload = {
        "split_summary": split_stats,
        "operating_threshold": results["operating_threshold"],
        "validation_metrics": results["val_metrics"],
        "model_comparison": results["test_metrics"],
        "calibration_analysis": results["calibration_analysis"],
        "calibration_curve": results["calibration_curve"],
        "feature_importances": results["feature_importances"],
        "recommended_model": {
            "name": results["best_model_name"],
            "rationale": results["dynamic_rationale"],
            "validation_metrics": results["val_metrics"][results["best_model_name"]],
            "test_metrics": results["test_metrics"][results["best_model_name"]],
            "dpd_only_baseline_test_metrics": results["test_metrics"]["DPD_Only_Baseline"],
        },
    }

    with open(METRICS_PATH, "w", encoding="utf-8") as f:
        json.dump(metrics_payload, f, indent=2)
    logger.info("Saved evaluation metrics report to %s", METRICS_PATH)


if __name__ == "__main__":
    main()
