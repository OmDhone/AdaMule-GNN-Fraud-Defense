"""Visualization of adversarial attack trajectories and topology modifications."""

from pathlib import Path
from typing import Any, Dict, List, Optional
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

from adamule.data.graph_builder import TransactionGraph


def plot_attack_trajectory(
    attack_history: List[Dict[str, Any]],
    target_node: int,
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """Plot detector fraud probability across perturbation steps."""
    steps = [0] + [h["step"] for h in attack_history]
    probs = [attack_history[0]["prev_prob"] if attack_history else 1.0] + [h["curr_prob"] for h in attack_history]

    fig, ax = plt.subplots(figsize=(7, 4.5), dpi=150)
    ax.plot(steps, probs, marker="o", color="#c0392b", linewidth=2.5, label="Detector Fraud Score")
    ax.axhline(0.5, color="#7f8c8d", linestyle="--", linewidth=1.5, label="Decision Threshold (0.50)")

    ax.set_xlabel("Perturbation Step", fontsize=11)
    ax.set_ylabel("Fraud Probability", fontsize=11)
    ax.set_title(f"Adversarial Evasion Trajectory: Node {target_node}", fontsize=12)
    ax.set_ylim(-0.05, 1.05)
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc="upper right", frameon=True)
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
        plt.close(fig)

    return fig


def plot_before_after_attack(
    graph_before: TransactionGraph,
    graph_after: TransactionGraph,
    target_node: int,
    save_path: Optional[str | Path] = None
) -> plt.Figure:
    """Side-by-side visualization of neighborhood topology before and after attack."""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5), dpi=150)

    # Subgraph before
    G1 = nx.DiGraph()
    e_before = graph_before.edge_index.cpu().numpy()
    out1 = e_before[1, e_before[0] == target_node]
    in1 = e_before[0, e_before[1] == target_node]
    for v in out1: G1.add_edge(target_node, int(v))
    for u in in1: G1.add_edge(int(u), target_node)

    pos1 = nx.spring_layout(G1, seed=42)
    nx.draw_networkx(G1, pos1, ax=ax1, node_color="#e74c3c", edge_color="#7f8c8d", node_size=400, font_size=8)
    ax1.set_title("Clean Topology (Before Attack)", fontsize=11)
    ax1.axis("off")

    # Subgraph after
    G2 = nx.DiGraph()
    e_after = graph_after.edge_index.cpu().numpy()
    out2 = e_after[1, e_after[0] == target_node]
    in2 = e_after[0, e_after[1] == target_node]
    for v in out2: G2.add_edge(target_node, int(v))
    for u in in2: G2.add_edge(int(u), target_node)

    pos2 = nx.spring_layout(G2, seed=42)
    nx.draw_networkx(G2, pos2, ax=ax2, node_color="#f39c12", edge_color="#2980b9", node_size=400, font_size=8)
    ax2.set_title("Evasive Topology (After Camouflage / Structuring)", fontsize=11)
    ax2.axis("off")

    plt.tight_layout()
    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
        plt.close(fig)

    return fig
