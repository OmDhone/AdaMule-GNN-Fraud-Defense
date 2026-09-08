"""Early stopping utility to prevent overfitting and capture optimal checkpoints."""

from pathlib import Path
from typing import Any, Dict, Optional
import torch


class EarlyStopping:
    """Tracks validation performance metric and saves the best model state."""

    def __init__(self, patience: int = 10, mode: str = "max", min_delta: float = 1e-4):
        self.patience = patience
        self.mode = mode
        self.min_delta = min_delta
        self.counter = 0
        self.best_score: Optional[float] = None
        self.early_stop = False
        self.best_state_dict: Optional[Dict[str, Any]] = None

    def __call__(self, score: float, model: torch.nn.Module) -> bool:
        """Update tracker with current epoch's validation metric.
        
        Returns:
            True if this is a new best model, False otherwise.
        """
        if self.best_score is None:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            return True

        if self.mode == "max":
            improved = score > self.best_score + self.min_delta
        else:
            improved = score < self.best_score - self.min_delta

        if improved:
            self.best_score = score
            self.best_state_dict = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            self.counter = 0
            return True
        else:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
            return False

    def load_best(self, model: torch.nn.Module) -> None:
        """Restore model weights to the best recorded state."""
        if self.best_state_dict is not None:
            model.load_state_dict(self.best_state_dict)
