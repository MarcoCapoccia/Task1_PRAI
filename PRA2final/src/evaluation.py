from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from sklearn import metrics


@dataclass
# stores classification metrics for reuse
class ClassificationResults:
    accuracy: float
    f1_macro: float
    f1_weighted: Optional[float]
    roc_auc_ovr: Optional[float]


# computes the main supervised metrics
def evaluate_classification(
    y_true: Sequence[int],
    y_pred: Sequence[int],
    y_proba: Optional[np.ndarray] = None,
) -> ClassificationResults:

    y_true_arr = np.asarray(y_true)
    y_pred_arr = np.asarray(y_pred)

    acc = metrics.accuracy_score(y_true_arr, y_pred_arr)
    f1_macro = metrics.f1_score(y_true_arr, y_pred_arr, average="macro")
    f1_weighted = metrics.f1_score(y_true_arr, y_pred_arr, average="weighted")

    roc_auc = None
    # runs auc only when probabilities are available
    if y_proba is not None:
        try:
            roc_auc = metrics.roc_auc_score(
                y_true_arr, y_proba, multi_class="ovr", average="macro"
            )
        except Exception:
            roc_auc = None

    return ClassificationResults(
        accuracy=acc,
        f1_macro=f1_macro,
        f1_weighted=f1_weighted,
        roc_auc_ovr=roc_auc,
    )


# returns the confusion matrix for plotting
def compute_confusion_matrix(
    y_true: Sequence[int],
    y_pred: Sequence[int],
) -> np.ndarray:

    return metrics.confusion_matrix(y_true, y_pred)


# computes core clustering quality metrics
def clustering_scores(
    X: np.ndarray,
    labels_true: Sequence[int],
    labels_pred: Sequence[int],
) -> Dict[str, float]:

    labels_true_arr = np.asarray(labels_true)
    labels_pred_arr = np.asarray(labels_pred)

    # keeps silhouette robust when clusters are degenerate
    try:
        sil = metrics.silhouette_score(X, labels_pred_arr)
    except Exception:
        sil = np.nan

    nmi = metrics.normalized_mutual_info_score(labels_true_arr, labels_pred_arr)
    ami = metrics.adjusted_mutual_info_score(labels_true_arr, labels_pred_arr)

    return {
        "silhouette": float(sil),
        "nmi": float(nmi),
        "ami": float(ami),
    }