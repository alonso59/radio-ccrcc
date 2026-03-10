"""Classification metric helpers used by visualization code."""

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from scipy.integrate import trapezoid


def compute_confusion_matrix(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    num_classes: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute raw and row-normalized confusion matrices."""
    confusion = np.zeros((num_classes, num_classes), dtype=np.int64)
    np.add.at(confusion, (y_true.astype(int), y_pred.astype(int)), 1)

    with np.errstate(divide="ignore", invalid="ignore"):
        normalized = confusion / confusion.sum(axis=1, keepdims=True)
        normalized = np.nan_to_num(normalized)

    return confusion, normalized


def compute_binary_roc(y_true: np.ndarray, y_scores: np.ndarray) -> Dict[str, Any]:
    """Compute ROC curve arrays and summary metrics for binary classification."""
    scores = _binary_scores(y_scores)
    fpr, tpr, thresholds = _compute_roc_curve(y_true, scores)
    threshold, sensitivity, specificity = _find_optimal_threshold(fpr, tpr, thresholds)
    return {
        "fpr": fpr,
        "tpr": tpr,
        "thresholds": thresholds,
        "auc": float(trapezoid(tpr, fpr)),
        "threshold": threshold,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def compute_multiclass_ovr_roc(
    y_true: np.ndarray,
    y_scores: np.ndarray,
    class_names: Optional[List[str]] = None,
) -> Tuple[List[Dict[str, Any]], float]:
    """Compute one-vs-rest ROC curves for multiclass classification."""
    curves: List[Dict[str, Any]] = []

    for class_index in range(y_scores.shape[1]):
        y_binary = (y_true == class_index).astype(int)

        try:
            fpr, tpr, _ = _compute_roc_curve(y_binary, y_scores[:, class_index])
        except ValueError:
            continue

        auc = float(trapezoid(tpr, fpr))
        label = (
            class_names[class_index]
            if class_names is not None and class_index < len(class_names)
            else f"Class {class_index}"
        )
        curves.append({"label": label, "fpr": fpr, "tpr": tpr, "auc": auc})

    macro_auc = float(np.mean([curve["auc"] for curve in curves])) if curves else 0.0
    return curves, macro_auc


def _binary_scores(y_scores: np.ndarray) -> np.ndarray:
    return y_scores.reshape(-1) if y_scores.ndim == 1 else y_scores[:, -1]


def _compute_roc_curve(
    y_true: np.ndarray,
    y_score: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    order = np.argsort(-y_score)
    y_sorted = y_true[order].astype(np.float64)
    scores_sorted = y_score[order]

    n_pos = y_sorted.sum()
    n_neg = len(y_sorted) - n_pos
    if n_pos == 0 or n_neg == 0:
        raise ValueError("Cannot compute ROC curve with only one class present")

    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1.0 - y_sorted)
    tpr = tp / n_pos
    fpr = fp / n_neg

    return (
        np.concatenate([[0.0], fpr]),
        np.concatenate([[0.0], tpr]),
        np.concatenate([[np.inf], scores_sorted]),
    )


def _find_optimal_threshold(
    fpr: np.ndarray,
    tpr: np.ndarray,
    thresholds: np.ndarray,
) -> Tuple[float, float, float]:
    j_scores = tpr - fpr
    optimal_idx = int(np.argmax(j_scores))
    return (
        float(thresholds[optimal_idx]),
        float(tpr[optimal_idx]),
        float(1.0 - fpr[optimal_idx]),
    )
