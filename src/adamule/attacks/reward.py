"""Reward calculation engine for evasion attack simulation."""

from typing import Dict, Optional


class EvasionRewardCalculator:
    """Calculates evasion reward based on detector confidence drop and penalty terms."""

    def __init__(
        self,
        confidence_reduction_weight: float = 2.0,
        constraint_violation_penalty: float = 5.0,
        structural_unrealism_penalty: float = 1.0,
        step_cost: float = 0.05
    ):
        self.w_conf = confidence_reduction_weight
        self.pen_constraint = constraint_violation_penalty
        self.pen_unrealism = structural_unrealism_penalty
        self.step_cost = step_cost

    def compute_reward(
        self,
        prev_prob: float,
        curr_prob: float,
        is_valid: bool = True,
        structural_distortion: float = 0.0
    ) -> float:
        """Compute scalar reward for the current perturbation step.
        
        Args:
            prev_prob: Detector fraud probability before action.
            curr_prob: Detector fraud probability after action.
            is_valid: Whether the action passed domain constraint checks.
            structural_distortion: Degree of deviation from realistic business patterns.
            
        Returns:
            Scalar reward float.
        """
        confidence_drop = prev_prob - curr_prob
        reward = self.w_conf * confidence_drop - self.step_cost

        if not is_valid:
            reward -= self.pen_constraint

        if structural_distortion > 0.0:
            reward -= self.pen_unrealism * structural_distortion

        return float(reward)
