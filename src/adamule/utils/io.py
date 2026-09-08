"""I/O utilities for saving/loading checkpoints, configurations, and metrics."""

import json
from pathlib import Path
from typing import Any, Dict
import torch


def save_json(data: Dict[str, Any], file_path: str | Path, indent: int = 2) -> None:
    """Save dictionary data to a formatted JSON file."""
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=indent, default=str)


def load_json(file_path: str | Path) -> Dict[str, Any]:
    """Load JSON file into dictionary."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file does not exist: {path}")
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_checkpoint(
    state_dict: Dict[str, Any],
    file_path: str | Path,
    metadata: Dict[str, Any] | None = None
) -> None:
    """Save PyTorch checkpoint with model weights and metadata.
    
    Args:
        state_dict: Model state_dict.
        file_path: Path to target .pt file.
        metadata: Optional dictionary with metrics, epoch, hyperparameters.
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "state_dict": state_dict,
        "metadata": metadata or {}
    }
    torch.save(payload, str(path))


def load_checkpoint(file_path: str | Path, device: str = "cpu") -> Dict[str, Any]:
    """Load PyTorch checkpoint and return dict with state_dict and metadata."""
    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {path}")
    return torch.load(str(path), map_location=device)
