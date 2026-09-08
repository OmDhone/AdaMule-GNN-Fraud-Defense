"""Graph topology visualization utilities for transaction and mule networks."""

from pathlib import Path
from typing import List, Optional, Set
import matplotlib.pyplot as plt
import networkx as nx
import torch

from adamule.data.graph_builder import TransactionGraph


def plot_subgraph(
    graph: TransactionGraph,
    center_node: int,
    hops: int = 2,
    max_neighbors: int = 30,
    save_path: Optional[str | Path] = None,
    title: Optional[str] = None
) -> plt.Figure:
    """Extract and plot the local ego-network around a center node."""
    G = nx.DiGraph()
    edge_index = graph.edge_index.cpu().numpy()
    E = edge_index.shape[1]

    # BFS to collect k-hop neighborhood
    visited: Set[int] = {center_node}
    frontier: Set[int] = {center_node}

    for _ in range(hops):
        next_frontier = set()
        for u in frontier:
            # Outgoing edges
            out_neighbors = edge_index[1, edge_index[0] == u]
            # Incoming edges
            in_neighbors = edge_index[0, edge_index[1] == u]
            for v in list(out_neighbors)[:max_neighbors]:
                G.add_edge(u, int(v))
                if v not in visited:
                    next_frontier.add(int(v))
                    visited.add(int(v))
            for v in list(in_neighbors)[:max_neighbors]:
                G.add_edge(int(v), u)
                if v not in visited:
                    next_frontier.add(int(v))
                    visited.add(int(v))
        frontier = next_frontier
        if len(visited) > max_neighbors * 2:
            break

    fig, ax = plt.subplots(figsize=(8, 6), dpi=150)
    pos = nx.spring_layout(G, seed=42)

    # Node colors: red if fraud, green if legitimate merchant, blue otherwise
    node_colors = []
    node_sizes = []
    y_np = graph.y.cpu().numpy()
    y_legit_np = graph.y_legitimacy.cpu().numpy()

    for node in G.nodes():
        if node == center_node:
            node_sizes.append(600)
        else:
            node_sizes.append(300)

        if y_np[node] == 1:
            node_colors.append("#e74c3c")  # Fraud Red
        elif y_legit_np[node] == 1:
            node_colors.append("#2ecc71")  # Legit Merchant Green
        else:
            node_colors.append("#3498db")  # Benign Retail Blue

    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_sizes, ax=ax, alpha=0.9)
    nx.draw_networkx_edges(G, pos, edge_color="#7f8c8d", arrowsize=12, width=1.0, ax=ax, alpha=0.6)
    nx.draw_networkx_labels(G, pos, font_size=8, font_family="sans-serif", ax=ax)

    ax.set_title(title or f"Local Subgraph: Node {center_node} ({'Fraud' if y_np[center_node] == 1 else 'Benign'})", fontsize=12)
    ax.axis("off")
    plt.tight_layout()

    if save_path:
        p = Path(save_path)
        p.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(p, bbox_inches="tight")
        plt.close(fig)

    return fig
