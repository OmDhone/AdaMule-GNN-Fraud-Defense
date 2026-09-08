"""Robustness analysis across perturbation budgets and attack algorithms."""

from typing import Any, Dict, List
import pandas as pd


class RobustnessAnalyzer:
    """Analyzes and compiles model degradation across multiple attack regimes."""

    @staticmethod
    def build_summary_table(experiment_records: List[Dict[str, Any]]) -> pd.DataFrame:
        """Constructs a clean comparison table as specified in Section 26.
        
        Columns:
            Model | Clean Recall | Attack Recall | Recall Drop | Robustness Ratio | FPR | Hard-Negative FPR | F1
        """
        rows = []
        for rec in experiment_records:
            rows.append({
                "Model / Setting": rec.get("model_name", "Unknown"),
                "Attack Method": rec.get("attack_method", "None"),
                "Clean Recall": rec.get("clean_recall", 0.0),
                "Attack Recall": rec.get("adversarial_recall", 0.0),
                "Recall Drop": rec.get("recall_drop", 0.0),
                "Robustness Ratio": rec.get("robustness_ratio", 0.0),
                "FPR": rec.get("adversarial_fpr", rec.get("clean_fpr", 0.0)),
                "Hard-Neg FPR": rec.get("hard_negative_fpr", 0.0),
                "F1-Score": rec.get("adversarial_f1", rec.get("clean_f1", 0.0))
            })
        return pd.DataFrame(rows)
