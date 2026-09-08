"""Gymnasium-compatible structuring attack environment for reinforcement learning."""

from typing import Any, Dict, Optional, Tuple
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import torch
import torch.nn as nn

from adamule.attacks.actions import ActionApplier
from adamule.attacks.constraints import FinancialConstraintsEngine
from adamule.attacks.reward import EvasionRewardCalculator
from adamule.data.graph_builder import TransactionGraph


class StructuringAttackEnv(gym.Env):
    """Gymnasium environment for training adaptive money-mule structuring attackers."""

    metadata = {"render_modes": ["human"]}

    def __init__(
        self,
        graph: TransactionGraph,
        detector: nn.Module,
        target_nodes: Optional[list] = None,
        max_steps: int = 12,
        seed: int = 42
    ):
        super().__init__()
        self.original_graph = graph
        self.detector = detector
        self.detector.eval()
        self.max_steps = max_steps
        self.constraints = FinancialConstraintsEngine()
        self.reward_calc = EvasionRewardCalculator()
        self.rng = np.random.default_rng(seed)

        # Target nodes: fraudulent nodes
        if target_nodes is None:
            fraud_idxs = (graph.y == 1).nonzero(as_tuple=True)[0].cpu().numpy()
            self.target_nodes = list(fraud_idxs) if len(fraud_idxs) > 0 else [0]
        else:
            self.target_nodes = list(target_nodes)

        # Legitimate candidate merchants
        merch_nodes = (graph.y_legitimacy == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        self.legit_merchants = list(merch_nodes) if len(merch_nodes) > 0 else list(range(min(15, graph.num_nodes)))

        # Action Space:
        # 0: Smurf-split outgoing transfer
        # 1: Add camouflage edge to high-degree merchant
        # 2: Add micro-purchase to second merchant
        # 3: Dilation time-shift
        # 4: Terminate attack
        self.action_space = spaces.Discrete(5)

        # Observation Space:
        # [fraud_prob, node_in_deg, node_out_deg, fan_in_ratio, budget_remaining, step_ratio, has_business]
        self.observation_space = spaces.Box(
            low=0.0,
            high=1.0,
            shape=(7,),
            dtype=np.float32
        )

        self.current_graph: Optional[TransactionGraph] = None
        self.current_target: int = 0
        self.current_step: int = 0
        self.current_prob: float = 1.0
        self.edges_added: int = 0
        self.edges_modified: int = 0

    def _get_detector_prob(self) -> float:
        with torch.no_grad():
            m_name = self.detector.__class__.__name__.lower()
            if "adamule" in m_name or "temporal" in m_name:
                l = self.detector(self.current_graph.x, self.current_graph.edge_index, self.current_graph.edge_attr, self.current_graph.edge_time)
            else:
                l = self.detector(self.current_graph.x, self.current_graph.edge_index)
            return float(torch.sigmoid(l[self.current_target]).item())

    def _get_obs(self) -> np.ndarray:
        in_deg = float((self.current_graph.edge_index[1] == self.current_target).sum().item())
        out_deg = float((self.current_graph.edge_index[0] == self.current_target).sum().item())
        total = in_deg + out_deg
        fan_in = in_deg / (total + 1e-5)
        budget_left = 1.0 - (self.edges_added + self.edges_modified) / float(self.constraints.max_steps)
        step_ratio = self.current_step / float(self.max_steps)
        has_business = float(self.current_graph.y_legitimacy[self.current_target].item())

        obs = np.array([
            self.current_prob,
            min(1.0, in_deg / 20.0),
            min(1.0, out_deg / 20.0),
            fan_in,
            max(0.0, budget_left),
            step_ratio,
            has_business
        ], dtype=np.float32)
        return obs

    def reset(self, seed: Optional[int] = None, options: Optional[Dict[str, Any]] = None) -> Tuple[np.ndarray, Dict[str, Any]]:
        super().reset(seed=seed)
        if seed is not None:
            self.rng = np.random.default_rng(seed)

        self.current_graph = self.original_graph.clone()
        self.current_target = int(self.rng.choice(self.target_nodes))
        self.current_step = 0
        self.edges_added = 0
        self.edges_modified = 0
        self.current_prob = self._get_detector_prob()

        obs = self._get_obs()
        return obs, {"target_node": self.current_target, "initial_prob": self.current_prob}

    def step(self, action: int) -> Tuple[np.ndarray, float, bool, bool, Dict[str, Any]]:
        self.current_step += 1
        prev_prob = self.current_prob
        is_valid = True
        info = {"action": action, "valid": True}

        # Check step budget
        budget_ok, _ = self.constraints.validate_budget(self.current_step, self.edges_added, self.edges_modified)
        if not budget_ok:
            terminated = True
            reward = -1.0
            return self._get_obs(), reward, terminated, False, info

        if action == 4:
            # Voluntary termination
            terminated = True
            reward = self.reward_calc.compute_reward(prev_prob, prev_prob, is_valid=True)
            return self._get_obs(), reward, terminated, False, info

        elif action == 0:
            # Smurf split
            out_edges = (self.current_graph.edge_index[0] == self.current_target).nonzero(as_tuple=True)[0].cpu().numpy()
            if len(out_edges) > 0:
                e_choice = int(self.rng.choice(out_edges))
                orig_log_amt = self.current_graph.edge_attr[e_choice, 0].item()
                orig_amt = float(np.expm1(orig_log_amt))
                if orig_amt > 150.0:
                    splits = [orig_amt * 0.33, orig_amt * 0.33, orig_amt * 0.34]
                    times = [100.0, 300.0, 600.0]
                    v_split, _ = self.constraints.validate_structuring_split(orig_amt, splits)
                    if v_split:
                        e_idx, e_attr, e_time = ActionApplier.apply_smurf_split(
                            self.current_graph.edge_index,
                            self.current_graph.edge_attr,
                            self.current_graph.edge_time,
                            e_choice,
                            splits,
                            times
                        )
                        self.current_graph.edge_index = e_idx
                        self.current_graph.edge_attr = e_attr
                        self.current_graph.edge_time = e_time
                        self.edges_modified += 1
                    else:
                        is_valid = False
                else:
                    is_valid = False
            else:
                is_valid = False

        elif action in [1, 2]:
            # Merchant camouflage edge
            m_target = int(self.rng.choice(self.legit_merchants))
            amt = float(self.rng.uniform(80.0, 350.0))
            ts = float(self.current_graph.edge_time.max().item() + self.rng.uniform(60, 1800))
            v_amt, _ = self.constraints.validate_amount(amt)
            if v_amt:
                e_idx, e_attr, e_time = ActionApplier.apply_add_camouflage_edge(
                    self.current_graph.edge_index,
                    self.current_graph.edge_attr,
                    self.current_graph.edge_time,
                    sender=self.current_target,
                    merchant_receiver=m_target,
                    amount=amt,
                    timestamp=ts,
                    edge_dim=self.current_graph.edge_attr.shape[1]
                )
                self.current_graph.edge_index = e_idx
                self.current_graph.edge_attr = e_attr
                self.current_graph.edge_time = e_time
                self.edges_added += 1
            else:
                is_valid = False

        elif action == 3:
            # Time shift
            out_edges = (self.current_graph.edge_index[0] == self.current_target).nonzero(as_tuple=True)[0].cpu().numpy()
            if len(out_edges) > 0:
                e_choice = int(self.rng.choice(out_edges))
                self.current_graph.edge_time[e_choice] += 3600.0  # delay by 1h
                self.edges_modified += 1
            else:
                is_valid = False

        self.current_prob = self._get_detector_prob()
        reward = self.reward_calc.compute_reward(prev_prob, self.current_prob, is_valid=is_valid)

        # Terminated if target evaded (<0.40) or budget exhausted
        terminated = self.current_prob < 0.40 or self.current_step >= self.max_steps
        truncated = False
        info["valid"] = is_valid
        info["curr_prob"] = self.current_prob

        return self._get_obs(), reward, terminated, truncated, info
