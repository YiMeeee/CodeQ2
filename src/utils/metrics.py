from __future__ import annotations

from typing import Dict

import numpy as np
from sklearn.metrics import accuracy_score, cohen_kappa_score, f1_score, recall_score, roc_auc_score


def _safe_auc(y_true: np.ndarray, y_prob: np.ndarray) -> float:
    try:
        return float(roc_auc_score(y_true, y_prob))
    except ValueError:
        return 0.5


def _softmax_np(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x, axis=1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.clip(exp_x.sum(axis=1, keepdims=True), a_min=1e-8, a_max=None)


def compute_metrics(
    grade_true: np.ndarray,
    grade_logits: np.ndarray,
    surgery_true: np.ndarray,
    surgery_prob: np.ndarray,
) -> Dict[str, float]:
    grade_pred = grade_logits.argmax(axis=1)
    grade_prob = _softmax_np(grade_logits)

    acc = accuracy_score(grade_true, grade_pred)
    f1_macro = f1_score(grade_true, grade_pred, average="macro")
    qwk = cohen_kappa_score(grade_true, grade_pred, weights="quadratic")
    recall_g3 = recall_score(
        (grade_true == 3).astype(int),
        (grade_pred == 3).astype(int),
        zero_division=0,
    )

    try:
        grade_auc_ovr = float(
            roc_auc_score(grade_true, grade_prob, multi_class="ovr", average="macro")
        )
    except ValueError:
        grade_auc_ovr = 0.5

    surgery_auc = _safe_auc(surgery_true, surgery_prob)

    return {
        "grade_acc": float(acc),
        "grade_f1_macro": float(f1_macro),
        "grade_qwk": float(qwk),
        "grade_auc_ovr": float(grade_auc_ovr),
        "grade_recall_g3": float(recall_g3),
        "surgery_auc": float(surgery_auc),
    }
