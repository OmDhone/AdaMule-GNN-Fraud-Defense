#!/usr/bin/env python3
"""CLI script to evaluate model checkpoints on clean and adversarial graphs.

Usage:
    python scripts/evaluate.py --model gcn --checkpoint outputs/models/baseline_gcn/best_model.pt
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adamule.data.dataset import TransactionGraphDataset
from adamule.evaluation.metrics import compute_classification_metrics
from adamule.models.gcn import GCN
from adamule.models.gat import GAT
from adamule.models.care_gnn import CAREGNN
from adamule.models.adamule import AdaMule
from adamule.utils.config import get_project_root
from adamule.utils.io import load_checkpoint, save_json
from adamule.utils.logging import get_logger

logger = get_logger("scripts.evaluate")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate trained graph fraud detector checkpoints.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "care_gnn", "adamule"])
    parser.add_argument("--checkpoint", type=str, required=True, help="Path to model checkpoint (.pt).")
    parser.add_argument("--profile", type=str, default="development")
    parser.add_argument("--graph-path", type=str, default=None, help="Optional custom graph path (e.g. evasive graph).")
    args = parser.parse_args()

    # Load graph
    if args.graph_path:
        logger.info(f"Loading custom graph from: {args.graph_path}")
        graph = torch.load(args.graph_path, weights_only=False)
    else:
        dataset = TransactionGraphDataset(profile=args.profile)
        graph = dataset.process()

    in_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]

    # Instantiate model
    if args.model == "gcn":
        model = GCN(in_dim=in_dim, hidden_dim=64)
    elif args.model == "gat":
        model = GAT(in_dim=in_dim, hidden_dim=64)
    elif args.model == "care_gnn":
        model = CAREGNN(in_dim=in_dim, hidden_dim=64)
    else:
        model = AdaMule(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=64)

    ckpt = load_checkpoint(args.checkpoint)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    with torch.no_grad():
        m_name = model.__class__.__name__.lower()
        if "adamule" in m_name or "temporal" in m_name:
            logits = model(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
        elif "caregnn" in m_name or "fraudre" in m_name:
            logits = model(graph.x, graph.edge_index, aux_edges=graph.aux_edges)
        else:
            logits = model(graph.x, graph.edge_index)

    test_mask = graph.test_mask
    y_test = graph.y[test_mask]
    y_prob = torch.sigmoid(logits[test_mask])
    hard_neg_mask = (graph.y_legitimacy[test_mask] == 1) & (y_test == 0)

    metrics = compute_classification_metrics(y_test, y_prob, threshold=0.5, hard_negative_mask=hard_neg_mask)

    logger.info("================ Evaluation Results ================")
    for k, v in metrics.items():
        logger.info(f"  {k:30s}: {v}")
    logger.info("====================================================")


if __name__ == "__main__":
    main()
