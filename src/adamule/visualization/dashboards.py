"""Interactive local Streamlit research dashboard for AdaMule.

Implements all 5 sections from Section 40 and Explainability from Section 41:
1. Overview
2. Graph Explorer
3. Fraud Detection & Interpretable Signals
4. Attack Simulation
5. Robustness Comparison
"""

import sys
from pathlib import Path
import streamlit as st
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt

# Ensure src is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent.parent))

from adamule.data.dataset import TransactionGraphDataset
from adamule.models.gcn import GCN
from adamule.models.adamule import AdaMule
from adamule.attacks.heuristic import HeuristicAttacker
from adamule.visualization.graph_plots import plot_subgraph
from adamule.visualization.attack_plots import plot_attack_trajectory
from adamule.utils.config import get_project_root
from adamule.utils.io import load_checkpoint, load_json


@st.cache_resource
def load_data_and_models():
    dataset = TransactionGraphDataset(profile="development")
    graph = dataset.process()

    in_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]

    # Load GCN
    gcn = GCN(in_dim=in_dim, hidden_dim=64)
    gcn_ckpt = get_project_root() / "outputs" / "models" / "baseline_gcn" / "best_model.pt"
    if gcn_ckpt.exists():
        gcn.load_state_dict(load_checkpoint(gcn_ckpt)["state_dict"])
    gcn.eval()

    # Load AdaMule
    adamule = AdaMule(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=64)
    adamule_ckpt = get_project_root() / "outputs" / "models" / "adamule" / "best_model.pt"
    if adamule_ckpt.exists():
        adamule.load_state_dict(load_checkpoint(adamule_ckpt)["state_dict"])
    adamule.eval()

    return graph, gcn, adamule


def main():
    st.set_page_config(
        page_title="AdaMule: Graph Fraud & Robustness Dashboard",
        layout="wide",
        initial_sidebar_state="expanded"
    )

    st.title("🛡️ AdaMule: Adversarially Robust Graph Fraud Detection")
    st.caption("Research prototype for detecting structuring-evasive money mule networks and preserving legitimate business activity.")

    graph, gcn, adamule = load_data_and_models()

    sidebar_selection = st.sidebar.radio(
        "Navigation",
        ["Overview", "Graph Explorer", "Fraud Detection & Explainability", "Attack Simulation", "Robustness Benchmark"]
    )

    # 1. OVERVIEW
    if sidebar_selection == "Overview":
        st.header("📊 Ecosystem Overview")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Accounts", f"{graph.num_nodes:,}")
        col2.metric("Total Transactions", f"{graph.num_edges:,}")
        col3.metric("Fraud Nodes Identified", f"{int(graph.y.sum().item()):,}")
        col4.metric("Legitimate Merchants", f"{int(graph.y_legitimacy.sum().item()):,}")

        st.subheader("Architectural Paradigm")
        st.markdown("""
        **AdaMule** addresses the critical vulnerability of payment Graph Neural Networks to **evasion attacks**:
        - **Structuring (Smurfing)**: Splitting illicit flows into fragments below standard reporting thresholds.
        - **Camouflage**: Fraud rings inserting benign transactions to high-degree legitimate merchant aggregators.
        - **Legitimacy Preservation**: Preventing legitimate small businesses with bursty seasonal flows from being wrongly flagged as fraud rings (Hard Negatives).
        """)

    # 2. GRAPH EXPLORER
    elif sidebar_selection == "Graph Explorer":
        st.header("🔍 Transaction Graph Explorer")
        st.write("Inspect local ego-networks for any account in the payment ecosystem.")

        node_id = st.number_input("Select Account Index", min_value=0, max_value=graph.num_nodes - 1, value=0, step=1)
        hops = st.slider("Neighborhood Hops", min_value=1, max_value=3, value=2)

        fig = plot_subgraph(graph, center_node=node_id, hops=hops)
        st.pyplot(fig)
        plt.close(fig)

    # 3. FRAUD DETECTION & EXPLAINABILITY
    elif sidebar_selection == "Fraud Detection & Explainability":
        st.header("🔎 Node Risk Profiler & Explainability Signals")
        node_id = st.number_input("Target Account Index", min_value=0, max_value=graph.num_nodes - 1, value=1, step=1)

        with torch.no_grad():
            gcn_prob = float(torch.sigmoid(gcn(graph.x, graph.edge_index)[node_id]).item())
            ada_prob = float(torch.sigmoid(adamule(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)[node_id]).item())

        c1, c2 = st.columns(2)
        c1.metric("Baseline GCN Fraud Probability", f"{gcn_prob:.3f}")
        c2.metric("AdaMule Fraud Probability", f"{ada_prob:.3f}")

        st.subheader("Interpretable Signals (Section 41)")
        in_deg = int((graph.edge_index[1] == node_id).sum().item())
        out_deg = int((graph.edge_index[0] == node_id).sum().item())
        is_legit = bool(graph.y_legitimacy[node_id].item() == 1)

        signals = []
        if in_deg > 10 and out_deg > 5:
            signals.append("⚠️ High fan-in and rapid fan-out aggregation (Potential Mule Aggregator or Merchant)")
        if is_legit:
            signals.append("✅ Verified Legitimate Business Profile (Protected Hard Negative)")
        if in_deg > 5 and out_deg <= 1:
            signals.append("⚠️ Fan-in accumulation without commercial profile")
        if in_deg == 0 and out_deg > 5:
            signals.append("⚠️ Rapid fan-out distribution from source account")

        if not signals:
            signals.append("ℹ️ Standard retail consumer transaction patterns")

        for s in signals:
            st.info(s)

    # 4. ATTACK SIMULATION
    elif sidebar_selection == "Attack Simulation":
        st.header("⚡ Adversarial Structuring Attack Simulator")
        st.write("Simulate an adaptive fraudster attempting to evade graph detection within domain financial constraints.")

        fraud_nodes = (graph.y == 1).nonzero(as_tuple=True)[0].cpu().numpy()
        target_node = st.selectbox("Select Fraudulent Target Account", options=list(fraud_nodes))
        method = st.selectbox("Attack Strategy", options=["heuristic", "random"])
        budget = st.slider("Perturbation Budget (Max Steps)", min_value=2, max_value=15, value=8)

        if st.button("Execute Attack Simulation"):
            attacker = HeuristicAttacker(detector=gcn)
            evasive_graph, summary = attacker.attack_node(graph, target_node=int(target_node), method=method, max_steps=budget)

            st.success(f"Attack Simulation Finished! Target Evaded: {summary['evaded']}")
            col1, col2, col3 = st.columns(3)
            col1.metric("Initial Fraud Prob", f"{summary['initial_prob']:.3f}")
            col2.metric("Final Evasive Prob", f"{summary['final_prob']:.3f}")
            col3.metric("Steps Executed", summary["steps_taken"])

            if summary["history"]:
                fig = plot_attack_trajectory(summary["history"], target_node=int(target_node))
                st.pyplot(fig)
                plt.close(fig)

                st.subheader("Step-by-Step Perturbation Log")
                st.dataframe(pd.DataFrame(summary["history"]))

    # 5. ROBUSTNESS BENCHMARK
    elif sidebar_selection == "Robustness Benchmark":
        st.header("🏆 Baseline vs. AdaMule Robustness Comparison")
        st.write("Live measured experimental metrics on clean and adversarially restructured test graphs.")

        comp_path = get_project_root() / "outputs" / "reports" / "benchmark_comparison.csv"
        if comp_path.exists():
            df_comp = pd.read_csv(comp_path)
            st.dataframe(df_comp, use_container_width=True)
        else:
            st.info("Run the automated experiment pipeline `python scripts/run_experiment.py` to populate all benchmark records.")


if __name__ == "__main__":
    main()
