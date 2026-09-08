"""Heuristic and Random evasion attackers for financial transaction graphs.

Implements:
- Stage 1: Random valid perturbations (validation of attack space)
- Stage 2: Heuristic structuring and camouflage attacks (amount smurfing and merchant disguise)
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import torch
import torch.nn as nn

from adamule.attacks.actions import ActionApplier, GraphPerturbationAction
from adamule.attacks.constraints import FinancialConstraintsEngine
from adamule.attacks.reward import EvasionRewardCalculator
from adamule.data.graph_builder import TransactionGraph
from adamule.utils.logging import get_logger

logger = get_logger("adamule.attacks.heuristic")


class HeuristicAttacker:
    """Performs heuristic domain-constrained evasion attacks against graph fraud detectors."""

    def __init__(
        self,
        detector: nn.Module,
        constraints_engine: Optional[FinancialConstraintsEngine] = None,
        reward_calculator: Optional[EvasionRewardCalculator] = None,
        seed: int = 42
    ):
        self.detector = detector
        self.detector.eval()
        self.constraints = constraints_engine or FinancialConstraintsEngine()
        self.reward_calc = reward_calculator or EvasionRewardCalculator()
        self.rng = np.random.default_rng(seed)

    @torch.no_grad()
    def _evaluate_node_prob(self, graph: TransactionGraph, node_idx: int) -> float:
        """Query detector fraud probability for a specific node."""
        m_name = self.detector.__class__.__name__.lower()
        if "adamule" in m_name or "temporal" in m_name:
            logits = self.detector(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
        elif "caregnn" in m_name or "fraudre" in m_name:
            logits = self.detector(graph.x, graph.edge_index, aux_edges=graph.aux_edges)
        else:
            logits = self.detector(graph.x, graph.edge_index)
        prob = torch.sigmoid(logits[node_idx]).item()
        return float(prob)

    def attack_node(
        self,
        graph: TransactionGraph,
        target_node: int,
        method: str = "heuristic",
        max_steps: int = 10,
        unconstrained: bool = False
    ) -> Tuple[TransactionGraph, Dict[str, Any]]:
        """Run iterative attack on a fraudulent node to evade detection.
        
        Args:
            graph: Original TransactionGraph.
            target_node: Integer index of fraudulent account to disguise.
            method: 'random' (Stage 1) or 'heuristic' (Stage 2).
            max_steps: Maximum perturbation steps.
            unconstrained: If True (for ablation Study I), bypass domain constraints.
            
        Returns:
            Tuple of (perturbed_graph, attack_log).
        """
        evasive_graph = graph.clone()
        initial_prob = self._evaluate_node_prob(evasive_graph, target_node)
        curr_prob = initial_prob

        history = []
        edges_added = 0
        edges_modified = 0

        # Identify candidate legitimate merchants in the graph (high degree, y_legitimacy == 1)
        legit_merchants = (evasive_graph.y_legitimacy == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        if len(legit_merchants) == 0:
            legit_merchants = np.arange(min(20, evasive_graph.num_nodes))

        # Find edges involving the target node
        for step in range(max_steps):
            if curr_prob < 0.40:  # Successfully evaded detector
                break

            # Check budget
            if not unconstrained:
                budget_ok, b_msg = self.constraints.validate_budget(step, edges_added, edges_modified)
                if not budget_ok:
                    break

            prev_prob = curr_prob
            action_taken = "none"

            if method == "random":
                # Stage 1: Random valid camouflage edge
                m_target = int(self.rng.choice(legit_merchants))
                amount = float(self.rng.uniform(50.0, 500.0))
                ts = float(evasive_graph.edge_time.max().item() + self.rng.uniform(60, 3600))
                valid, msg = self.constraints.validate_amount(amount)
                if valid or unconstrained:
                    e_idx, e_attr, e_time = ActionApplier.apply_add_camouflage_edge(
                        evasive_graph.edge_index,
                        evasive_graph.edge_attr,
                        evasive_graph.edge_time,
                        sender=target_node,
                        merchant_receiver=m_target,
                        amount=amount,
                        timestamp=ts,
                        edge_dim=evasive_graph.edge_attr.shape[1]
                    )
                    evasive_graph.edge_index = e_idx
                    evasive_graph.edge_attr = e_attr
                    evasive_graph.edge_time = e_time
                    edges_added += 1
                    action_taken = "random_camouflage_edge"

            else:
                # Stage 2: Heuristic structuring and merchant disguise
                # Check if target has outgoing transactions that can be structured (split)
                out_edges = (evasive_graph.edge_index[0] == target_node).nonzero(as_tuple=True)[0].cpu().numpy()

                if len(out_edges) > 0 and step % 2 == 0 and edges_modified < self.constraints.max_edges_modified:
                    # Smurfing / Structuring split
                    e_choice = int(self.rng.choice(out_edges))
                    # Check original amount from log(amount)
                    orig_log_amt = evasive_graph.edge_attr[e_choice, 0].item()
                    orig_amount = float(np.expm1(orig_log_amt))

                    if orig_amount > 200.0:
                        splits = [round(orig_amount / 3.0, 2), round(orig_amount / 3.0, 2), round(orig_amount - 2 * round(orig_amount / 3.0, 2), 2)]
                        times = [60.0, 300.0, 600.0]
                        valid_split, split_msg = self.constraints.validate_structuring_split(orig_amount, splits)
                        if valid_split or unconstrained:
                            e_idx, e_attr, e_time = ActionApplier.apply_smurf_split(
                                evasive_graph.edge_index,
                                evasive_graph.edge_attr,
                                evasive_graph.edge_time,
                                edge_idx_to_split=e_choice,
                                split_amounts=splits,
                                time_increments=times
                            )
                            evasive_graph.edge_index = e_idx
                            evasive_graph.edge_attr = e_attr
                            evasive_graph.edge_time = e_time
                            edges_modified += 1
                            action_taken = "smurf_split"
                else:
                    # Disguise with legitimate merchant camouflage edge
                    m_target = int(self.rng.choice(legit_merchants))
                    amount = float(self.rng.uniform(100.0, 450.0))
                    ts = float(evasive_graph.edge_time.max().item() + self.rng.uniform(300, 1800))
                    valid, _ = self.constraints.validate_amount(amount)
                    if valid or unconstrained:
                        e_idx, e_attr, e_time = ActionApplier.apply_add_camouflage_edge(
                            evasive_graph.edge_index,
                            evasive_graph.edge_attr,
                            evasive_graph.edge_time,
                            sender=target_node,
                            merchant_receiver=m_target,
                            amount=amount,
                            timestamp=ts,
                            edge_dim=evasive_graph.edge_attr.shape[1]
                        )
                        evasive_graph.edge_index = e_idx
                        evasive_graph.edge_attr = e_attr
                        evasive_graph.edge_time = e_time
                        edges_added += 1
                        action_taken = "merchant_camouflage_edge"

            curr_prob = self._evaluate_node_prob(evasive_graph, target_node)
            step_reward = self.reward_calc.compute_reward(prev_prob, curr_prob, is_valid=True)

            history.append({
                "step": step + 1,
                "action": action_taken,
                "prev_prob": round(prev_prob, 4),
                "curr_prob": round(curr_prob, 4),
                "reward": round(step_reward, 4)
            })

        evaded = curr_prob < 0.50
        summary = {
            "target_node": target_node,
            "initial_prob": round(initial_prob, 4),
            "final_prob": round(curr_prob, 4),
            "prob_reduction": round(initial_prob - curr_prob, 4),
            "evaded": evaded,
            "steps_taken": len(history),
            "edges_added": edges_added,
            "edges_modified": edges_modified,
            "history": history
        }
        return evasive_graph, summary
