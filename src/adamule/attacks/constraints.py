"""Financial constraint verification engine for adversarial graph perturbations.

Ensures every proposed transaction graph modification conforms to real-world financial rules:
- Transaction amount bounds [min_amount, max_amount]
- Velocity / frequency limits (hourly, daily)
- Temporal plausibility (causality: timestamps must not backtrack before receipt of funds)
- Balance / fund conservation
- Maximum perturbation budget (edges added, modified, total steps)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch


class FinancialConstraintsEngine:
    """Validates candidate perturbations against domain financial rules."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        cfg = config or {}
        constraint_cfg = cfg.get("constraints", {})
        budget_cfg = cfg.get("budget", {})

        self.min_amount = float(constraint_cfg.get("min_transaction_amount", 10.0))
        self.max_amount = float(constraint_cfg.get("max_transaction_amount", 100000.0))
        self.max_hourly_freq = int(constraint_cfg.get("max_hourly_frequency", 10))
        self.max_daily_freq = int(constraint_cfg.get("max_daily_frequency", 50))
        self.amount_tolerance = float(constraint_cfg.get("amount_preservation_tolerance", 0.05))
        self.min_gap_seconds = float(constraint_cfg.get("min_inter_arrival_seconds", 60.0))

        # Perturbation Budget
        self.max_steps = int(budget_cfg.get("max_steps", 15))
        self.max_edges_added = int(budget_cfg.get("max_edges_added", 8))
        self.max_edges_modified = int(budget_cfg.get("max_edges_modified", 10))
        self.max_amount_shift = float(budget_cfg.get("max_amount_shift", 0.20))

    def validate_amount(self, amount: float) -> Tuple[bool, str]:
        """Verify transaction amount is within plausible financial thresholds."""
        if amount < self.min_amount:
            return False, f"Amount {amount:.2f} is below minimum allowed ({self.min_amount})"
        if amount > self.max_amount:
            return False, f"Amount {amount:.2f} exceeds maximum allowed ({self.max_amount})"
        return True, "Valid amount"

    def validate_temporal_causality(
        self,
        new_timestamp: float,
        preceding_timestamp: Optional[float] = None
    ) -> Tuple[bool, str]:
        """Verify new transaction does not violate temporal order (e.g. forward before receipt)."""
        if preceding_timestamp is not None and new_timestamp < preceding_timestamp:
            return False, f"Temporal violation: transfer timestamp {new_timestamp} is earlier than fund receipt {preceding_timestamp}"
        return True, "Valid timing"

    def validate_budget(
        self,
        current_step: int,
        edges_added: int,
        edges_modified: int
    ) -> Tuple[bool, str]:
        """Verify attacker has not exhausted the allowable perturbation budget."""
        if current_step >= self.max_steps:
            return False, f"Exceeded maximum steps budget ({self.max_steps})"
        if edges_added >= self.max_edges_added:
            return False, f"Exceeded maximum added edges budget ({self.max_edges_added})"
        if edges_modified >= self.max_edges_modified:
            return False, f"Exceeded maximum modified edges budget ({self.max_edges_modified})"
        return True, "Within budget"

    def validate_structuring_split(
        self,
        original_amount: float,
        split_amounts: List[float]
    ) -> Tuple[bool, str]:
        """Verify fund conservation when splitting an amount into structured fragments."""
        total_split = sum(split_amounts)
        diff = abs(total_split - original_amount)
        if diff > (original_amount * self.amount_tolerance):
            return False, f"Fund conservation violated: original {original_amount}, sum of splits {total_split}"
        for a in split_amounts:
            valid, msg = self.validate_amount(a)
            if not valid:
                return False, msg
        return True, "Valid structuring split"
