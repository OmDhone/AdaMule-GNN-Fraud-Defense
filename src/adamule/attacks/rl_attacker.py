"""Stage 4: Reinforcement learning attacker for adaptive structuring evasion.

Trains a PPO or Policy-based agent in StructuringAttackEnv against a frozen detector.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

try:
    from stable_baselines3 import PPO
    HAS_SB3 = True
except ImportError:
    HAS_SB3 = False

from adamule.attacks.environment import StructuringAttackEnv
from adamule.data.graph_builder import TransactionGraph
from adamule.utils.logging import get_logger

logger = get_logger("adamule.attacks.rl")


class RLStructuringAttacker:
    """Trains and executes an RL agent to find evasive transaction structures."""

    def __init__(
        self,
        detector: nn.Module,
        output_dir: Optional[str | Path] = None,
        seed: int = 42
    ):
        self.detector = detector
        self.detector.eval()
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/models/rl_attacker")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed
        self.model = None

    def train(
        self,
        graph: TransactionGraph,
        total_timesteps: int = 1500
    ) -> Any:
        """Train PPO attacker inside StructuringAttackEnv."""
        env = StructuringAttackEnv(graph=graph, detector=self.detector, seed=self.seed)

        if HAS_SB3:
            logger.info(f"Training PPO RL Attacker for {total_timesteps} timesteps...")
            self.model = PPO(
                policy="MlpPolicy",
                env=env,
                learning_rate=3e-4,
                n_steps=64,
                batch_size=32,
                gamma=0.99,
                seed=self.seed,
                verbose=0
            )
            self.model.learn(total_timesteps=total_timesteps)
            save_path = self.output_dir / "ppo_attacker.zip"
            self.model.save(str(save_path))
            logger.info(f"Saved trained PPO policy to: {save_path}")
        else:
            logger.info("Stable-Baselines3 not available; using heuristic policy.")
        return self.model

    def attack_node(
        self,
        graph: TransactionGraph,
        target_node: int,
        max_steps: int = 10
    ) -> Tuple[TransactionGraph, Dict[str, Any]]:
        """Deploy trained policy to disguise a target fraudulent node."""
        env = StructuringAttackEnv(
            graph=graph,
            detector=self.detector,
            target_nodes=[target_node],
            max_steps=max_steps,
            seed=self.seed
        )
        obs, info = env.reset()
        initial_prob = info["initial_prob"]
        curr_prob = initial_prob
        history = []

        terminated = False
        step = 0

        while not terminated and step < max_steps:
            step += 1
            if self.model is not None and HAS_SB3:
                action, _ = self.model.predict(obs, deterministic=True)
                action = int(action)
            else:
                # Heuristic action choice
                action = 1 if step % 2 == 1 else 0

            obs, reward, terminated, truncated, step_info = env.step(action)
            curr_prob = step_info.get("curr_prob", curr_prob)
            history.append({
                "step": step,
                "action": action,
                "curr_prob": round(curr_prob, 4),
                "reward": round(reward, 4),
                "valid": step_info.get("valid", True)
            })

        summary = {
            "target_node": target_node,
            "initial_prob": round(initial_prob, 4),
            "final_prob": round(curr_prob, 4),
            "prob_reduction": round(initial_prob - curr_prob, 4),
            "evaded": curr_prob < 0.50,
            "steps_taken": len(history),
            "edges_added": env.edges_added,
            "edges_modified": env.edges_modified,
            "history": history
        }
        return env.current_graph, summary
