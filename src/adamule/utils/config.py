"""Configuration management utilities for AdaMule."""

import os
from pathlib import Path
from typing import Any, Dict, Optional
import yaml


def get_project_root() -> Path:
    """Return the absolute path to the project root directory."""
    # This file is located at <project_root>/src/adamule/utils/config.py
    return Path(__file__).resolve().parent.parent.parent.parent


def load_config(config_path: str | Path) -> Dict[str, Any]:
    """Load a YAML configuration file.
    
    Args:
        config_path: Path to YAML configuration file, either relative to project root or absolute.
        
    Returns:
        Dictionary with parsed configuration values.
    """
    path = Path(config_path)
    if not path.is_absolute():
        path = get_project_root() / path
        
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
        
    with open(path, "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)
    return config or {}


def get_profile_config(profile_name: str = "development", data_config_path: Optional[str | Path] = None) -> Dict[str, Any]:
    """Retrieve data configuration for a specific profile (development, medium, research).
    
    Args:
        profile_name: Name of profile ('development', 'medium', 'research').
        data_config_path: Optional custom path to data.yaml.
        
    Returns:
        Dictionary of configuration values for that profile.
    """
    if data_config_path is None:
        data_config_path = get_project_root() / "configs" / "data.yaml"
        
    config = load_config(data_config_path)
    profiles = config.get("profiles", {})
    if profile_name not in profiles:
        available = list(profiles.keys())
        raise KeyError(f"Profile '{profile_name}' not found. Available profiles: {available}")
        
    profile_cfg = profiles[profile_name]
    profile_cfg["profile_name"] = profile_name
    return profile_cfg
