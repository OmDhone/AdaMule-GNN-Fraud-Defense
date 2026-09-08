#!/usr/bin/env python3
"""CLI script to execute evasion attacks and evaluate detector degradation.

Usage:
    python scripts/run_attack.py --model gcn --method heuristic --profile development
"""

import argparse
import sys
from pathlib import Path
import torch
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adamule.attacks.heuristic import HeuristicAttacker
from adamule.attacks.gradient_attack import ConstrainedGradientAttacker
from adamule.attacks.rl_attacker import RLStructuringAttacker
from adamule.data.dataset import TransactionGraphDataset
from adamule.evaluation.metrics import compute_classification_metrics, compute_robustness_metrics
from adamule.models.gcn import GCN
from adamule.models.gat import GAT
from adamule.models.care_gnn import CAREGNN
from adamule.models.fraudre import FRAUDRE
from adamule.models.adamule import AdaMule
from adamule.utils.config import get_project_root
from adamule.utils.io import load_checkpoint, save_json
from adamule.utils.logging import get_logger
from adamule.utils.seed import set_seed

logger = get_logger("scripts.run_attack")


def load_detector_model(model_name: str, in_dim: int, edge_dim: int) -> torch.nn.Module:
    root = get_project_root()
    if model_name == "gcn":
        model = GCN(in_dim=in_dim, hidden_dim=64)
        ckpt_dir = root / "outputs" / "models" / "baseline_gcn"
    elif model_name == "gat":
        model = GAT(in_dim=in_dim, hidden_dim=64)
        ckpt_dir = root / "outputs" / "models" / "baseline_gat"
    elif model_name == "care_gnn":
        model = CAREGNN(in_dim=in_dim, hidden_dim=64)
        ckpt_dir = root / "outputs" / "models" / "baseline_care_gnn"
    elif model_name == "fraudre":
        model = FRAUDRE(in_dim=in_dim, hidden_dim=64)
        ckpt_dir = root / "outputs" / "models" / "baseline_fraudre"
    else:
        model = AdaMule(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=64)
        ckpt_dir = root / "outputs" / "models" / "adamule"

    ckpt_file = ckpt_dir / "best_model.pt"
    if ckpt_file.exists():
        ckpt = load_checkpoint(ckpt_file)
        model.load_state_dict(ckpt["state_dict"])
        logger.info(f"Loaded weights from {ckpt_file}")
    else:
        logger.warning(f"No checkpoint found at {ckpt_file}; using un-trained model for evaluation demo.")
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser(
        description="Execute evasion attack against a trained graph fraud detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "care_gnn", "fraudre", "adamule"])
    parser.add_argument("--method", type=str, default="heuristic", choices=["random", "heuristic", "gradient", "rl"])
    parser.add_argument("--profile", type=str, default="development")
    parser.add_argument("--max-steps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--unconstrained", action="store_true", help="Bypass constraints (Ablation I).")
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Running Attack: {args.method.upper()} on Model: {args.model.upper()} | Profile: {args.profile}")

    dataset = TransactionGraphDataset(profile=args.profile, seed=args.seed)
    graph = dataset.process()

    detector = load_detector_model(args.model, graph.x.shape[1], graph.edge_attr.shape[1])

    # Clean Evaluation on test set
    with torch.no_grad():
        m_name = detector.__class__.__name__.lower()
        if "adamule" in m_name or "temporal" in m_name:
            logits = detector(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
        elif "caregnn" in m_name or "fraudre" in m_name:
            logits = detector(graph.x, graph.edge_index, aux_edges=graph.aux_edges)
        else:
            logits = detector(graph.x, graph.edge_index)

    test_mask = graph.test_mask
    y_test = graph.y[test_mask]
    prob_clean = torch.sigmoid(logits[test_mask])
    hard_neg_mask = (graph.y_legitimacy[test_mask] == 1) & (y_test == 0)

    clean_metrics = compute_classification_metrics(y_test, prob_clean, hard_negative_mask=hard_neg_mask)
    logger.info(f"Clean Performance -> Recall: {clean_metrics['recall']}, F1: {clean_metrics['f1']}, FPR: {clean_metrics['fpr']}")

    # Identify fraud test nodes to attack
    fraud_test_nodes = (graph.test_mask & (graph.y == 1)).nonzero(as_tuple=True)[0].cpu().numpy()
    if len(fraud_test_nodes) == 0:
        fraud_test_nodes = (graph.y == 1).nonzero(as_tuple=True)[0].cpu().numpy()

    logger.info(f"Launching attack against {len(fraud_test_nodes)} fraudulent nodes...")

    # Attacker dispatch
    if args.method in ["random", "heuristic"]:
        attacker = HeuristicAttacker(detector=detector, seed=args.seed)
    elif args.method == "gradient":
        attacker = ConstrainedGradientAttacker(detector=detector)
    else:
        attacker = RLStructuringAttacker(detector=detector, seed=args.seed)

    evasive_graph = graph.clone()
    attack_logs = []
    evaded_count = 0

    for node in fraud_test_nodes:
        if args.method in ["random", "heuristic"]:
            evasive_graph, summary = attacker.attack_node(
                evasive_graph,
                target_node=int(node),
                method=args.method,
                max_steps=args.max_steps,
                unconstrained=args.unconstrained
            )
        elif args.method == "gradient":
            evasive_graph, summary = attacker.attack_node(
                evasive_graph,
                target_node=int(node),
                max_steps=args.max_steps
            )
        else:
            evasive_graph, summary = attacker.attack_node(
                evasive_graph,
                target_node=int(node),
                max_steps=args.max_steps
            )

        if summary["evaded"]:
            evaded_count += 1
        attack_logs.append(summary)

    # Re-evaluate detector on attacked graph
    with torch.no_grad():
        if "adamule" in m_name or "temporal" in m_name:
            logits_adv = detector(evasive_graph.x, evasive_graph.edge_index, evasive_graph.edge_attr, evasive_graph.edge_time)
        elif "caregnn" in m_name or "fraudre" in m_name:
            logits_adv = detector(evasive_graph.x, evasive_graph.edge_index, aux_edges=evasive_graph.aux_edges)
        else:
            logits_adv = detector(evasive_graph.x, evasive_graph.edge_index)

    prob_adv = torch.sigmoid(logits_adv[test_mask])
    attack_metrics = compute_classification_metrics(y_test, prob_adv, hard_negative_mask=hard_neg_mask)
    robustness_results = compute_robustness_metrics(clean_metrics, attack_metrics)

    asr = (evaded_count / len(fraud_test_nodes)) if len(fraud_test_nodes) > 0 else 0.0
    robustness_results["attack_success_rate"] = round(asr, 4)
    robustness_results["total_attacked_nodes"] = len(fraud_test_nodes)
    robustness_results["evaded_nodes"] = evaded_count

    logger.info(
        f"Attack Results -> Clean Recall: {robustness_results['clean_recall']}, "
        f"Attack Recall: {robustness_results['adversarial_recall']}, "
        f"Recall Drop: {robustness_results['recall_drop']}, "
        f"Attack Success Rate (ASR): {robustness_results['attack_success_rate'] * 100:.1f}%"
    )

    # Save results
    out_dir = get_project_root() / "experiments" / "attacks" / f"{args.model}_{args.method}"
    out_dir.mkdir(parents=True, exist_ok=True)
    save_json(robustness_results, out_dir / "attack_summary.json")
    save_json({"logs": attack_logs}, out_dir / "attack_logs.json")
    torch.save(evasive_graph, out_dir / "evasive_graph.pt")
    logger.info(f"Saved attack outputs and evasive graph to: {out_dir}")


if __name__ == "__main__":
    main()
