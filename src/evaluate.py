import numpy as np
import pandas as pd

from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)


def calculate_metrics(y_true, y_proba, threshold: float = 0.5) -> dict:
    """Calculate classification metrics for a selected threshold."""
    y_pred = (y_proba >= threshold).astype(int)

    return {
        "threshold": threshold,
        "roc_auc": roc_auc_score(y_true, y_proba),
        "pr_auc": average_precision_score(y_true, y_proba),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "confusion_matrix": confusion_matrix(y_true, y_pred).tolist(),
    }


def make_model_comparison_row(
    model_name: str,
    y_true,
    y_proba,
    threshold: float = 0.5,
) -> dict:
    """Create one row for model comparison table."""
    metrics = calculate_metrics(y_true, y_proba, threshold)

    return {
        "model": model_name,
        "threshold": threshold,
        "roc_auc": metrics["roc_auc"],
        "pr_auc": metrics["pr_auc"],
        "precision": metrics["precision"],
        "recall": metrics["recall"],
        "f1": metrics["f1"],
    }


def threshold_search(
    y_true,
    y_proba,
    thresholds=None,
    false_positive_cost: int = 5,
    false_negative_cost: int = 100,
) -> pd.DataFrame:
    """
    Search decision threshold using precision, recall, F1 and business cost.

    Business cost:
        business_cost = FP_COST * FP + FN_COST * FN
    """
    if thresholds is None:
        thresholds = np.arange(0.01, 1.0, 0.01)

    rows = []

    for threshold in thresholds:
        y_pred = (y_proba >= threshold).astype(int)

        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()

        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)

        business_cost = (
            false_positive_cost * fp
            + false_negative_cost * fn
        )

        rows.append(
            {
                "threshold": round(float(threshold), 4),
                "precision": precision,
                "recall": recall,
                "f1": f1,
                "false_positives": int(fp),
                "false_negatives": int(fn),
                "true_positives": int(tp),
                "true_negatives": int(tn),
                "business_cost": int(business_cost),
            }
        )

    return pd.DataFrame(rows)