"""Evaluation metrics calculation for graph fraud detection and robustness."""

from typing import Any, Dict, Optional, Tuple
import numpy as np
from sklearn.metrics import (
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    confusion_matrix
)
import torch


def compute_classification_metrics(
    y_true: np.ndarray | torch.Tensor,
    y_prob: np.ndarray | torch.Tensor,
    threshold: float = 0.5,
    hard_negative_mask: Optional[np.ndarray | torch.Tensor] = None
) -> Dict[str, float]:
    """Compute comprehensive classification and fraud metrics.
    
    Args:
        y_true: Ground truth binary labels (0 = benign, 1 = fraud).
        y_prob: Predicted fraud probabilities [0, 1].
        threshold: Decision threshold.
        hard_negative_mask: Boolean mask indicating legitimate irregular merchants (hard negatives).
        
    Returns:
        Dictionary containing precision, recall, f1, roc_auc, pr_auc, fpr, fnr, and hard-negative metrics.
    """
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    if isinstance(y_prob, torch.Tensor):
        y_prob = y_prob.detach().cpu().numpy()

    y_pred = (y_prob >= threshold).astype(int)

    # Standard metrics
    prec = float(precision_score(y_true, y_pred, zero_division=0))
    rec = float(recall_score(y_true, y_pred, zero_division=0))
    f1 = float(f1_score(y_true, y_pred, zero_division=0))

    try:
        roc_auc = float(roc_auc_score(y_true, y_prob))
    except ValueError:
        roc_auc = 0.5

    try:
        pr_auc = float(average_precision_score(y_true, y_prob))
    except ValueError:
        pr_auc = 0.0

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    tn, fp, fn, tp = cm.ravel()

    fpr = float(fp / (fp + tn)) if (fp + tn) > 0 else 0.0
    fnr = float(fn / (fn + tp)) if (fn + tp) > 0 else 0.0

    metrics = {
        "precision": round(prec, 4),
        "recall": round(rec, 4),
        "f1": round(f1, 4),
        "roc_auc": round(roc_auc, 4),
        "pr_auc": round(pr_auc, 4),
        "fpr": round(fpr, 4),
        "fnr": round(fnr, 4),
        "true_positives": int(tp),
        "false_positives": int(fp),
        "true_negatives": int(tn),
        "false_negatives": int(fn),
    }

    # Hard-negative evaluation (Section 25: False Positive Rate on Legitimate Irregular Merchants)
    if hard_negative_mask is not None:
        if isinstance(hard_negative_mask, torch.Tensor):
            hard_negative_mask = hard_negative_mask.detach().cpu().numpy()
        hn_total = int(hard_negative_mask.sum())
        if hn_total > 0:
            hn_pred = y_pred[hard_negative_mask]
            hn_fp = int((hn_pred == 1).sum())
            hn_fpr = float(hn_fp / hn_total)
            metrics["hard_negative_total"] = hn_total
            metrics["hard_negative_false_positives"] = hn_fp
            metrics["hard_negative_fpr"] = round(hn_fpr, 4)
        else:
            metrics["hard_negative_fpr"] = 0.0

    return metrics


def compute_robustness_metrics(
    clean_metrics: Dict[str, float],
    attack_metrics: Dict[str, float]
) -> Dict[str, float]:
    """Compute robustness degradation metrics.
    
    Formulae:
        clean_recall = R_clean
        adversarial_recall = R_attack
        recall_drop = R_clean - R_attack
        robustness_ratio = R_attack / (R_clean + 1e-6)
    """
    r_clean = clean_metrics.get("recall", 0.0)
    r_attack = attack_metrics.get("recall", 0.0)
    drop = max(0.0, r_clean - r_attack)
    ratio = r_attack / (r_clean + 1e-6)

    return {
        "clean_recall": round(r_clean, 4),
        "adversarial_recall": round(r_attack, 4),
        "recall_drop": round(drop, 4),
        "robustness_ratio": round(min(1.0, ratio), 4),
        "clean_f1": clean_metrics.get("f1", 0.0),
        "adversarial_f1": attack_metrics.get("f1", 0.0),
        "clean_fpr": clean_metrics.get("fpr", 0.0),
        "adversarial_fpr": attack_metrics.get("fpr", 0.0),
        "hard_negative_fpr": attack_metrics.get("hard_negative_fpr", 0.0)
    }
