"""Model comparison and ablation benchmarking utilities."""

from typing import Any, Dict, List
import pandas as pd


def generate_experiment_comparison_table(results_dict: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """Generate standardized benchmark comparison table across experiments A through K."""
    rows = []
    for exp_id, res in results_dict.items():
        name = res.get("name", exp_id)
        metrics = res.get("metrics", {})
        rows.append({
            "Experiment": exp_id.upper(),
            "Description": name,
            "Clean Recall": metrics.get("clean_recall", metrics.get("recall", 0.0)),
            "Attack Recall": metrics.get("adversarial_recall", metrics.get("recall", 0.0)),
            "Recall Drop": metrics.get("recall_drop", 0.0),
            "F1-Score": metrics.get("f1", metrics.get("clean_f1", 0.0)),
            "PR-AUC": metrics.get("pr_auc", 0.0),
            "Overall FPR": metrics.get("fpr", 0.0),
            "Hard-Negative FPR": metrics.get("hard_negative_fpr", 0.0)
        })
    df = pd.DataFrame(rows)
    return df
