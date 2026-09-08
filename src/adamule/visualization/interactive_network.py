"""Interactive physics-based network visualization using PyVis and vis.js."""

from pathlib import Path
from typing import Optional, Set
import networkx as nx
from pyvis.network import Network
import torch

from adamule.data.graph_builder import TransactionGraph


def build_interactive_pyvis_html(
    graph: TransactionGraph,
    center_node: Optional[int] = None,
    hops: int = 2,
    max_nodes: int = 45,
    height: str = "600px",
    width: str = "100%"
) -> str:
    """Generate an interactive HTML string with force-directed physics and rich tooltips."""
    net = Network(height=height, width=width, bgcolor="#0f172a", font_color="#e2e8f0", directed=True)
    net.barnes_hut(gravity=-3000, central_gravity=0.3, spring_length=120, spring_strength=0.05, damping=0.95)

    edge_index = graph.edge_index.cpu().numpy()
    y_np = graph.y.cpu().numpy()
    y_legit_np = graph.y_legitimacy.cpu().numpy()

    # Determine nodes to display
    if center_node is not None:
        nodes_to_include: Set[int] = {center_node}
        frontier: Set[int] = {center_node}
        for _ in range(hops):
            nxt = set()
            for u in frontier:
                outs = edge_index[1, edge_index[0] == u]
                ins = edge_index[0, edge_index[1] == u]
                for v in outs:
                    nodes_to_include.add(int(v))
                    nxt.add(int(v))
                for v in ins:
                    nodes_to_include.add(int(v))
                    nxt.add(int(v))
            frontier = nxt
            if len(nodes_to_include) >= max_nodes:
                break
        selected_nodes = list(nodes_to_include)[:max_nodes]
    else:
        # Default: sample balanced set of fraud, merchants, and normal accounts
        fraud_nodes = list((graph.y == 1).nonzero(as_tuple=True)[0].cpu().numpy()[:15])
        merchant_nodes = list((graph.y_legitimacy == 1).nonzero(as_tuple=True)[0].cpu().numpy()[:15])
        normal_nodes = list(range(min(15, graph.num_nodes)))
        selected_nodes = list(set(fraud_nodes + merchant_nodes + normal_nodes))[:max_nodes]

    node_set = set(selected_nodes)

    # Add Nodes
    for n in selected_nodes:
        is_fraud = bool(y_np[n] == 1)
        is_merchant = bool(y_legit_np[n] == 1)

        acc_name = graph.account_ids[n] if n < len(graph.account_ids) else f"ACC_{n:05d}"
        in_deg = int((edge_index[1] == n).sum())
        out_deg = int((edge_index[0] == n).sum())

        if is_fraud:
            color = "#ef4444"  # Neon red
            role = "🔴 Money Mule Suspect"
            size = 26 if n == center_node else 20
        elif is_merchant:
            color = "#10b981"  # Emerald green
            role = "🟢 Legitimate Merchant Aggregator"
            size = 28 if n == center_node else 22
        else:
            color = "#3b82f6"  # Blue
            role = "🔵 Retail Individual Account"
            size = 22 if n == center_node else 16

        tooltip = f"""
        <div style='font-family: sans-serif; font-size: 13px; padding: 6px;'>
            <b>{acc_name}</b><br>
            <b>Classification:</b> {role}<br>
            <b>Incoming Transfers:</b> {in_deg}<br>
            <b>Outgoing Transfers:</b> {out_deg}<br>
            <b>Status:</b> {'⚠️ HIGH RISK FLAG' if is_fraud else '✅ Verified'}
        </div>
        """

        label = f"{acc_name}\n({in_deg} in / {out_deg} out)"
        net.add_node(
            n,
            label=label,
            title=tooltip,
            color=color,
            size=size,
            borderWidth=3 if n == center_node else 1,
            shape="dot"
        )

    # Add Edges
    for i in range(edge_index.shape[1]):
        src, dst = int(edge_index[0, i]), int(edge_index[1, i])
        if src in node_set and dst in node_set:
            is_fraud_edge = (y_np[src] == 1) and (y_np[dst] == 1)
            edge_color = "#f87171" if is_fraud_edge else "#64748b"
            net.add_edge(
                src,
                dst,
                color=edge_color,
                arrows="to",
                width=2.5 if is_fraud_edge else 1.2,
                smooth={"type": "curvedCW", "roundness": 0.15}
            )

    html_content = net.generate_html()
    return html_content
