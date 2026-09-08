"""Test suite verifying attack algorithms, Gymnasium environment, and reward calculations."""

import pytest
import torch
import numpy as np

from adamule.attacks.environment import StructuringAttackEnv
from adamule.attacks.heuristic import HeuristicAttacker
from adamule.attacks.gradient_attack import ConstrainedGradientAttacker
from adamule.attacks.reward import EvasionRewardCalculator
from adamule.data.dataset import TransactionGraphDataset
from adamule.models.gcn import GCN


@pytest.fixture
def trained_gcn_and_graph():
    dataset = TransactionGraphDataset(profile="development")
    graph = dataset.process()
    model = GCN(in_dim=graph.x.shape[1], hidden_dim=32, num_layers=2)
    model.eval()
    return model, graph


def test_evasion_reward_calculator():
    calc = EvasionRewardCalculator(confidence_reduction_weight=2.0, constraint_violation_penalty=5.0)
    
    # Positive reward when confidence drops (0.9 -> 0.4)
    r_pos = calc.compute_reward(prev_prob=0.9, curr_prob=0.4, is_valid=True)
    assert r_pos > 0.0
    
    # Negative penalty when action violates domain constraints
    r_invalid = calc.compute_reward(prev_prob=0.9, curr_prob=0.4, is_valid=False)
    assert r_invalid < 0.0


def test_heuristic_attacker_stages(trained_gcn_and_graph):
    model, graph = trained_gcn_and_graph
    attacker = HeuristicAttacker(detector=model)
    
    # Pick a fraud target
    fraud_nodes = (graph.y == 1).nonzero(as_tuple=True)[0].cpu().numpy()
    assert len(fraud_nodes) > 0
    target = int(fraud_nodes[0])
    
    # Stage 1: Random valid attack
    graph_rand, summary_rand = attacker.attack_node(graph, target_node=target, method="random", max_steps=3)
    assert summary_rand["target_node"] == target
    assert len(summary_rand["history"]) > 0
    
    # Stage 2: Heuristic structuring attack
    graph_heur, summary_heur = attacker.attack_node(graph, target_node=target, method="heuristic", max_steps=5)
    assert summary_heur["steps_taken"] > 0
    assert summary_heur["final_prob"] <= summary_heur["initial_prob"] + 0.05


def test_gradient_attacker(trained_gcn_and_graph):
    model, graph = trained_gcn_and_graph
    grad_attacker = ConstrainedGradientAttacker(detector=model, num_iterations=3)
    
    fraud_nodes = (graph.y == 1).nonzero(as_tuple=True)[0].cpu().numpy()
    target = int(fraud_nodes[0])
    
    graph_grad, summary_grad = grad_attacker.attack_node(graph, target_node=target, max_steps=3)
    assert "initial_prob" in summary_grad
    assert "final_prob" in summary_grad


def test_gymnasium_structuring_env(trained_gcn_and_graph):
    model, graph = trained_gcn_and_graph
    env = StructuringAttackEnv(graph=graph, detector=model, max_steps=5)
    
    obs, info = env.reset()
    assert obs.shape == (7,)
    assert "target_node" in info
    
    # Take valid action (e.g. merchant camouflage = 1)
    obs_next, reward, terminated, truncated, step_info = env.step(1)
    assert obs_next.shape == (7,)
    assert isinstance(reward, float)
    assert isinstance(terminated, bool)
