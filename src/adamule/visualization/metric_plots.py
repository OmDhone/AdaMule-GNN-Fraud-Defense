"""Publication-quality performance and robustness metric plots."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import numpy as np


def plot_clean_vs_adversarial_recall(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """Grouped bar chart comparing Clean Recall vs Adversarial Recall across models."""
    models = list(results_dict.keys())
    clean_recalls = [results_dict[m].get("clean_recall", 0.0) for m in models]
    attack_recalls = [results_dict[m].get("adversarial_recall", 0.0) for m in models]

    x = np.arange(len(models))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    rects1 = ax.bar(x - width / 2, clean_recalls, width, label="Clean Recall", color="#2980b9", alpha=0.85)
    rects2 = ax.bar(x + width / 2, attack_recalls, width, label="Adversarial Recall", color="#e74c3c", alpha=0.85)

    ax.set_ylabel("Detection Recall", fontsize=11)
    ax.set_title("Fraud Detector Robustness: Clean vs. Adversarially Restructured Fraud", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_hard_negative_fpr(
    results_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """Bar chart comparing False Positive Rate on Legitimate Irregular Merchants."""
    models = list(results_dict.keys())
    hn_fprs = [results_dict[m].get("hard_negative_fpr", 0.0) for m in models]

    fig, ax = plt.subplots(figsize=(8, 4.5), dpi=150)
    bars = ax.bar(models, hn_fprs, color="#e67e22", alpha=0.85, width=0.5)

    ax.set_ylabel("False Positive Rate (Hard Negatives)", fontsize=11)
    ax.set_title("Legitimate-vs-Camouflage Discrimination (Lower is Better)", fontsize=12)
    ax.set_xticks(np.arange(len(models)))
    ax.set_xticklabels(models, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.1)
    ax.grid(axis="y", linestyle=":", alpha=0.6)

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_ablation_study(
    ablation_dict: Dict[str, Dict[str, Any]],
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """Bar chart illustrating ablation findings across key architectural components."""
    labels = list(ablation_dict.keys())
    recalls = [ablation_dict[k].get("adversarial_recall", 0.0) for k in labels]
    fprs = [ablation_dict[k].get("hard_negative_fpr", 0.0) for k in labels]

    x = np.arange(len(labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(9, 5), dpi=150)
    ax.bar(x - width / 2, recalls, width, label="Adversarial Recall", color="#27ae60", alpha=0.85)
    ax.bar(x + width / 2, fprs, width, label="Hard-Negative FPR", color="#c0392b", alpha=0.85)

    ax.set_ylabel("Metric Value", fontsize=11)
    ax.set_title("Ablation Study: Contributions of Robust Training & Legitimacy Regularizer", fontsize=12)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=15, ha="right", fontsize=9)
    ax.set_ylim(0, 1.15)
    ax.grid(axis="y", linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
        plt.close(fig)

    return fig
