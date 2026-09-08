"""Model calibration and Expected Calibration Error (ECE) computation."""

from typing import Dict, Tuple
import numpy as np


def compute_calibration_error(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    n_bins: int = 10
) -> Dict[str, float]:
    """Compute Expected Calibration Error (ECE) and Maximum Calibration Error (MCE)."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    bin_indices = np.digitize(y_prob, bins) - 1
    bin_indices = np.clip(bin_indices, 0, n_bins - 1)

    ece = 0.0
    mce = 0.0
    total_samples = len(y_true)

    for b in range(n_bins):
        mask = bin_indices == b
        bin_count = np.sum(mask)
        if bin_count > 0:
            bin_acc = np.mean(y_true[mask])
            bin_conf = np.mean(y_prob[mask])
            abs_diff = np.abs(bin_acc - bin_conf)
            ece += (bin_count / total_samples) * abs_diff
            mce = max(mce, abs_diff)

    return {
        "ece": round(float(ece), 4),
        "mce": round(float(mce), 4)
    }
