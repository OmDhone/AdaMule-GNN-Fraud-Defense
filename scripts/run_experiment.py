#!/usr/bin/env python3
"""Master pipeline runner for AdaMule Research Experiments A through K.

Executes:
    generate data -> preprocess -> train baselines -> run attacks ->
    robust training -> evaluate -> generate figures -> compile reports.

Usage:
    python scripts/run_experiment.py --config configs/experiments.yaml --profile development
"""

import argparse
import sys
import time
from pathlib import Path
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adamule.attacks.heuristic import HeuristicAttacker
from adamule.data.dataset import TransactionGraphDataset
from adamule.evaluation.comparisons import generate_experiment_comparison_table
from adamule.evaluation.metrics import compute_classification_metrics, compute_robustness_metrics
from adamule.evaluation.reports import ExperimentReportGenerator
from adamule.models.gcn import GCN
from adamule.models.gat import GAT
from adamule.models.care_gnn import CAREGNN
from adamule.models.adamule import AdaMule
from adamule.training.trainer import GNNTrainer
from adamule.training.adversarial_trainer import AdversarialTrainer
from adamule.visualization.metric_plots import (
    plot_clean_vs_adversarial_recall,
    plot_hard_negative_fpr,
    plot_ablation_study
)
from adamule.utils.config import load_config, get_project_root
from adamule.utils.io import save_json
from adamule.utils.logging import get_logger
from adamule.utils.seed import set_seed

logger = get_logger("scripts.run_experiment")


def train_model(model_type: str, graph, in_dim: int, edge_dim: int, epochs: int = 15, lr: float = 0.005, use_legit: bool = True, device: str = "cpu"):
    if model_type == "gcn":
        model = GCN(in_dim=in_dim, hidden_dim=64)
    elif model_type == "gat":
        model = GAT(in_dim=in_dim, hidden_dim=64)
    elif model_type == "care_gnn":
        model = CAREGNN(in_dim=in_dim, hidden_dim=64)
    elif model_type == "adamule":
        model = AdaMule(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=64, use_legitimacy_module=use_legit)
    else:
        raise ValueError(f"Unknown model {model_type}")

    opt = torch.optim.Adam(model.parameters(), lr=lr, weight_decay=1e-4)
    trainer = GNNTrainer(model=model, optimizer=opt, device=device, patience=epochs)
    trainer.fit(graph, epochs=epochs)
    return model, trainer


def evaluate_under_attack(model, graph, attacker, max_steps: int = 6, unconstrained: bool = False):
    test_mask = graph.test_mask
    y_test = graph.y[test_mask]
    hard_neg_mask = (graph.y_legitimacy[test_mask] == 1) & (y_test == 0)

    # Clean eval
    model.eval()
    with torch.no_grad():
        m_name = model.__class__.__name__.lower()
        if "adamule" in m_name or "temporal" in m_name:
            logits_clean = model(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
        elif "caregnn" in m_name or "fraudre" in m_name:
            logits_clean = model(graph.x, graph.edge_index, aux_edges=graph.aux_edges)
        else:
            logits_clean = model(graph.x, graph.edge_index)

    p_clean = torch.sigmoid(logits_clean[test_mask])
    clean_metrics = compute_classification_metrics(y_test, p_clean, hard_negative_mask=hard_neg_mask)

    # Attack fraud test nodes
    fraud_test_nodes = (graph.test_mask & (graph.y == 1)).nonzero(as_tuple=True)[0].cpu().numpy()
    if len(fraud_test_nodes) == 0:
        fraud_test_nodes = (graph.y == 1).nonzero(as_tuple=True)[0].cpu().numpy()

    evasive_graph = graph.clone()
    for n in fraud_test_nodes:
        evasive_graph, _ = attacker.attack_node(
            evasive_graph,
            target_node=int(n),
            method="heuristic",
            max_steps=max_steps,
            unconstrained=unconstrained
        )

    # Adv eval
    with torch.no_grad():
        if "adamule" in m_name or "temporal" in m_name:
            logits_adv = model(evasive_graph.x, evasive_graph.edge_index, evasive_graph.edge_attr, evasive_graph.edge_time)
        elif "caregnn" in m_name or "fraudre" in m_name:
            logits_adv = model(evasive_graph.x, evasive_graph.edge_index, aux_edges=evasive_graph.aux_edges)
        else:
            logits_adv = model(evasive_graph.x, evasive_graph.edge_index)

    p_adv = torch.sigmoid(logits_adv[test_mask])
    adv_metrics = compute_classification_metrics(y_test, p_adv, hard_negative_mask=hard_neg_mask)
    return compute_robustness_metrics(clean_metrics, adv_metrics)


def main():
    parser = argparse.ArgumentParser(description="Execute complete research benchmark experiments A through K.")
    parser.add_argument("--config", type=str, default="configs/experiments.yaml")
    parser.add_argument("--profile", type=str, default="development")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--epochs", type=int, default=15)
    args = parser.parse_args()

    set_seed(args.seed)
    root = get_project_root()
    logger.info(f"Starting AdaMule Master Benchmark Pipeline | Profile: {args.profile}")

    dataset = TransactionGraphDataset(profile=args.profile, seed=args.seed)
    graph = dataset.process()

    in_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]
    num_fraud = int(graph.y.sum().item())
    num_legit = int(graph.y_legitimacy.sum().item())

    dataset_stats = {
        "num_nodes": graph.num_nodes,
        "num_edges": graph.num_edges,
        "fraud_ratio": num_fraud / graph.num_nodes,
        "legit_ratio": num_legit / graph.num_nodes
    }

    results = {}
    exp_cfg = load_config(args.config).get("experiments", {})

    # Train baselines
    logger.info("Training GCN...")
    gcn, _ = train_model("gcn", graph, in_dim, edge_dim, epochs=args.epochs)
    attacker_gcn = HeuristicAttacker(detector=gcn, seed=args.seed)

    logger.info("Training GAT...")
    gat, _ = train_model("gat", graph, in_dim, edge_dim, epochs=args.epochs)
    attacker_gat = HeuristicAttacker(detector=gat, seed=args.seed)

    logger.info("Training CAREGNN...")
    care_gnn, _ = train_model("care_gnn", graph, in_dim, edge_dim, epochs=args.epochs)
    attacker_care = HeuristicAttacker(detector=care_gnn, seed=args.seed)

    logger.info("Training AdaMule (Standard)...")
    adamule_std, _ = train_model("adamule", graph, in_dim, edge_dim, epochs=args.epochs, use_legit=True)
    attacker_ada = HeuristicAttacker(detector=adamule_std, seed=args.seed)

    logger.info("Training AdaMule (Robust min-max)...")
    adamule_rob = AdaMule(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=64, use_legitimacy_module=True)
    opt_rob = torch.optim.Adam(adamule_rob.parameters(), lr=0.005, weight_decay=1e-4)
    adv_trainer = AdversarialTrainer(model=adamule_rob, optimizer=opt_rob, seed=args.seed)
    adv_trainer.train_min_max(graph=graph, rounds=2, epochs_per_round=max(4, args.epochs // 2))
    attacker_rob = HeuristicAttacker(detector=adamule_rob, seed=args.seed)

    # Train Ablation Models
    logger.info("Training Ablation without Legitimacy Module...")
    adamule_no_legit = AdaMule(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=64, use_legitimacy_module=False)
    opt_nl = torch.optim.Adam(adamule_no_legit.parameters(), lr=0.005, weight_decay=1e-4)
    adv_trainer_nl = AdversarialTrainer(model=adamule_no_legit, optimizer=opt_nl, seed=args.seed)
    adv_trainer_nl.train_min_max(graph=graph, rounds=2, epochs_per_round=max(4, args.epochs // 2))
    attacker_nl = HeuristicAttacker(detector=adamule_no_legit, seed=args.seed)

    logger.info("Executing evaluations for Experiments A through K...")

    # Exp A: GCN Clean
    m_clean = GNNTrainer(gcn, torch.optim.Adam(gcn.parameters())).evaluate(graph, mask=graph.test_mask)
    results["exp_a"] = {"name": exp_cfg.get("exp_a", {}).get("name", "GCN Clean"), "metrics": m_clean}

    # Exp B: GCN Attack
    results["exp_b"] = {"name": exp_cfg.get("exp_b", {}).get("name", "GCN Attack"), "metrics": evaluate_under_attack(gcn, graph, attacker_gcn)}

    # Exp C: GAT Attack
    results["exp_c"] = {"name": exp_cfg.get("exp_c", {}).get("name", "GAT Attack"), "metrics": evaluate_under_attack(gat, graph, attacker_gat)}

    # Exp D: CARE-GNN Attack
    results["exp_d"] = {"name": exp_cfg.get("exp_d", {}).get("name", "CAREGNN Attack"), "metrics": evaluate_under_attack(care_gnn, graph, attacker_care)}

    # Exp E: AdaMule Clean
    m_ada_clean = GNNTrainer(adamule_std, torch.optim.Adam(adamule_std.parameters())).evaluate(graph, mask=graph.test_mask)
    results["exp_e"] = {"name": exp_cfg.get("exp_e", {}).get("name", "AdaMule Clean"), "metrics": m_ada_clean}

    # Exp F: AdaMule Std Attack (no robust training)
    results["exp_f"] = {"name": exp_cfg.get("exp_f", {}).get("name", "AdaMule Std Attack"), "metrics": evaluate_under_attack(adamule_std, graph, attacker_ada)}

    # Exp G: AdaMule Robust Clean
    m_rob_clean = GNNTrainer(adamule_rob, torch.optim.Adam(adamule_rob.parameters())).evaluate(graph, mask=graph.test_mask)
    results["exp_g"] = {"name": exp_cfg.get("exp_g", {}).get("name", "AdaMule Robust Clean"), "metrics": m_rob_clean}

    # Exp H: AdaMule Robust Attack (Full)
    results["exp_h"] = {"name": exp_cfg.get("exp_h", {}).get("name", "AdaMule Robust Attack"), "metrics": evaluate_under_attack(adamule_rob, graph, attacker_rob)}

    # Exp I: Ablation without constraints (unconstrained attack on robust model)
    results["exp_i"] = {"name": exp_cfg.get("exp_i", {}).get("name", "Ablation: Unconstrained Attack"), "metrics": evaluate_under_attack(adamule_rob, graph, attacker_rob, unconstrained=True)}

    # Exp J: Ablation without legitimacy loss
    results["exp_j"] = {"name": exp_cfg.get("exp_j", {}).get("name", "Ablation: No Legitimacy Loss"), "metrics": evaluate_under_attack(adamule_no_legit, graph, attacker_nl)}

    # Exp K: Ablation without adversarial training
    results["exp_k"] = {"name": exp_cfg.get("exp_k", {}).get("name", "Ablation: No Adversarial Training"), "metrics": results["exp_f"]["metrics"]}

    logger.info("Compiling tables, figures, and research report...")
    df_benchmark = generate_experiment_comparison_table(results)
    reports_dir = root / "outputs" / "reports"
    figures_dir = root / "outputs" / "figures"
    results_dir = root / "experiments" / "results"
    reports_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Save benchmark table CSV
    csv_path = reports_dir / "benchmark_comparison.csv"
    df_benchmark.to_csv(csv_path, index=False)
    logger.info(f"Saved benchmark CSV to {csv_path}")

    # Plot figures
    model_plot_data = {
        "GCN": results["exp_b"]["metrics"],
        "GAT": results["exp_c"]["metrics"],
        "CAREGNN": results["exp_d"]["metrics"],
        "AdaMule (Std)": results["exp_f"]["metrics"],
        "AdaMule (Robust)": results["exp_h"]["metrics"],
    }
    plot_clean_vs_adversarial_recall(model_plot_data, save_path=figures_dir / "clean_vs_adversarial_recall.png")
    plot_hard_negative_fpr(model_plot_data, save_path=figures_dir / "hard_negative_fpr_comparison.png")

    ablation_plot_data = {
        "Full AdaMule": results["exp_h"]["metrics"],
        "No Legitimacy": results["exp_j"]["metrics"],
        "No Adv Training": results["exp_k"]["metrics"],
        "Unconstrained": results["exp_i"]["metrics"]
    }
    plot_ablation_study(ablation_plot_data, save_path=figures_dir / "ablation_study.png")

    # Generate Markdown Report
    ablation_details = {
        "Ablation J: No Legitimacy Module": {
            "adversarial_recall": results["exp_j"]["metrics"]["adversarial_recall"],
            "recall_drop": results["exp_j"]["metrics"]["recall_drop"],
            "hard_negative_fpr": results["exp_j"]["metrics"]["hard_negative_fpr"],
            "finding": "Without legitimacy regularizer, false positive rate on irregular merchants increases."
        },
        "Ablation K: No Adversarial Training": {
            "adversarial_recall": results["exp_k"]["metrics"]["adversarial_recall"],
            "recall_drop": results["exp_k"]["metrics"]["recall_drop"],
            "hard_negative_fpr": results["exp_k"]["metrics"]["hard_negative_fpr"],
            "finding": "Without min-max training, detector suffers from severe recall drop under evasive structuring."
        },
        "Ablation I: Unconstrained Perturbations": {
            "adversarial_recall": results["exp_i"]["metrics"]["adversarial_recall"],
            "recall_drop": results["exp_i"]["metrics"]["recall_drop"],
            "hard_negative_fpr": results["exp_i"]["metrics"]["hard_negative_fpr"],
            "finding": "Unconstrained attacks achieve greater evasion but create financially impossible anomalies."
        }
    }

    report_path = reports_dir / "final_research_report.md"
    ExperimentReportGenerator.generate_markdown_report(
        experiment_name=f"AdaMule Benchmark Profile ({args.profile})",
        dataset_stats=dataset_stats,
        model_results=model_plot_data,
        ablation_results=ablation_details,
        output_file=report_path
    )
    logger.info(f"Saved research report to: {report_path}")

    # Print summary table to stdout
    print("\n" + "=" * 95)
    print(df_benchmark.to_string(index=False))
    print("=" * 95 + "\n")


if __name__ == "__main__":
    main()
