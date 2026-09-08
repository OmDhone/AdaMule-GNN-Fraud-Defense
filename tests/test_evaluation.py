"""Test suite verifying classification and robustness evaluation metrics."""

import pytest
import numpy as np
import torch

from adamule.evaluation.metrics import compute_classification_metrics, compute_robustness_metrics


def test_compute_classification_metrics():
    y_true = np.array([1, 1, 0, 0, 1, 0])
    y_prob = np.array([0.9, 0.8, 0.1, 0.2, 0.4, 0.3])
    
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5)
    
    assert "precision" in metrics
    assert "recall" in metrics
    assert "f1" in metrics
    assert "roc_auc" in metrics
    assert "pr_auc" in metrics
    assert "fpr" in metrics
    assert "fnr" in metrics
    assert 0.0 <= metrics["precision"] <= 1.0
    assert 0.0 <= metrics["recall"] <= 1.0


def test_hard_negative_evaluation():
    y_true = np.array([1, 0, 0, 0])
    y_prob = np.array([0.9, 0.8, 0.2, 0.1])
    # Node index 1 is a hard negative (legitimate merchant) but falsely flagged
    hard_neg_mask = np.array([False, True, False, False])
    
    metrics = compute_classification_metrics(y_true, y_prob, threshold=0.5, hard_negative_mask=hard_neg_mask)
    assert "hard_negative_fpr" in metrics
    assert metrics["hard_negative_fpr"] == 1.0
    assert metrics["hard_negative_false_positives"] == 1


def test_compute_robustness_metrics():
    clean = {"recall": 0.90, "f1": 0.85, "fpr": 0.05}
    attack = {"recall": 0.60, "f1": 0.65, "fpr": 0.08, "hard_negative_fpr": 0.04}
    
    rob = compute_robustness_metrics(clean, attack)
    assert rob["clean_recall"] == 0.90
    assert rob["adversarial_recall"] == 0.60
    assert round(rob["recall_drop"], 2) == 0.30
    assert round(rob["robustness_ratio"], 2) == 0.67
