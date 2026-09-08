#!/usr/bin/env python3
"""CLI script to train baseline GNN fraud detectors (GCN, GAT, CAREGNN, FRAUDRE, TemporalGNN).

Usage:
    python scripts/train_baseline.py --model gcn --profile development --epochs 30
    python scripts/train_baseline.py --model gat --profile development --epochs 30
"""

import argparse
import sys
from pathlib import Path
import torch

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adamule.data.dataset import TransactionGraphDataset
from adamule.models.gcn import GCN
from adamule.models.gat import GAT
from adamule.models.care_gnn import CAREGNN
from adamule.models.fraudre import FRAUDRE
from adamule.models.temporal import TemporalGNN
from adamule.training.trainer import GNNTrainer
from adamule.utils.config import load_config, get_project_root
from adamule.utils.logging import get_logger
from adamule.utils.seed import set_seed

logger = get_logger("scripts.train_baseline")


def build_baseline_model(model_name: str, in_dim: int, edge_dim: int = 12, cfg: dict | None = None) -> torch.nn.Module:
    """Instantiate baseline model by name."""
    cfg = cfg or {}
    hidden_dim = cfg.get("hidden_dim", 64)
    num_layers = cfg.get("num_layers", 2)
    dropout = cfg.get("dropout", 0.2)

    name = model_name.lower()
    if name == "gcn":
        return GCN(in_dim=in_dim, hidden_dim=hidden_dim, num_layers=num_layers, dropout=dropout)
    elif name == "gat":
        num_heads = cfg.get("num_heads", 4)
        return GAT(in_dim=in_dim, hidden_dim=hidden_dim, num_heads=num_heads, num_layers=num_layers, dropout=dropout)
    elif name == "care_gnn" or name == "caregnn":
        return CAREGNN(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif name == "fraudre":
        return FRAUDRE(in_dim=in_dim, hidden_dim=hidden_dim, dropout=dropout)
    elif name == "temporal":
        return TemporalGNN(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=hidden_dim, dropout=dropout)
    else:
        raise ValueError(f"Unknown baseline model: '{model_name}'. Choose from: gcn, gat, care_gnn, fraudre, temporal")


def main():
    parser = argparse.ArgumentParser(
        description="Train baseline graph fraud detection models.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "care_gnn", "fraudre", "temporal"], help="Model architecture.")
    parser.add_argument("--profile", type=str, default="development", choices=["development", "medium", "research"], help="Dataset scale profile.")
    parser.add_argument("--epochs", type=int, default=35, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu", help="Compute device ('cpu' or 'cuda').")
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Training Baseline Model: {args.model.upper()} | Profile: {args.profile} | Seed: {args.seed}")

    # Load dataset
    dataset = TransactionGraphDataset(profile=args.profile, seed=args.seed)
    graph = dataset.process()

    in_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]

    # Model configuration
    model_cfg = load_config("configs/model.yaml").get(args.model, {})
    model = build_baseline_model(args.model, in_dim=in_dim, edge_dim=edge_dim, cfg=model_cfg)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_dir = get_project_root() / "outputs" / "models" / f"baseline_{args.model}"

    trainer = GNNTrainer(
        model=model,
        optimizer=optimizer,
        device=args.device,
        patience=10,
        output_dir=out_dir
    )

    results = trainer.fit(graph=graph, epochs=args.epochs)
    logger.info(f"Completed baseline {args.model} training! Best Val F1: {results['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()
