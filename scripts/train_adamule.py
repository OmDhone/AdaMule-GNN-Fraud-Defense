#!/usr/bin/env python3
"""CLI script to train the AdaMule primary model with legitimacy module and robust training.

Usage:
    python scripts/train_adamule.py --profile development --epochs 30
    python scripts/train_adamule.py --profile development --robust --rounds 3
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adamule.data.dataset import TransactionGraphDataset
from adamule.models.adamule import AdaMule
from adamule.training.trainer import GNNTrainer
from adamule.training.adversarial_trainer import AdversarialTrainer
from adamule.utils.config import load_config, get_project_root
from adamule.utils.logging import get_logger
from adamule.utils.seed import set_seed

logger = get_logger("scripts.train_adamule")


def main():
    parser = argparse.ArgumentParser(
        description="Train the primary AdaMule model.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--profile", type=str, default="development", choices=["development", "medium", "research"])
    parser.add_argument("--epochs", type=int, default=30, help="Number of training epochs.")
    parser.add_argument("--lr", type=float, default=0.005, help="Learning rate.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--no-legitimacy", action="store_true", help="Disable auxiliary legitimacy module (Ablation J).")
    parser.add_argument("--robust", action="store_true", help="Enable min-max alternating robust adversarial training.")
    parser.add_argument("--rounds", type=int, default=3, help="Number of adversarial training rounds.")
    parser.add_argument("--attack-method", type=str, default="heuristic", choices=["heuristic", "gradient"])
    args = parser.parse_args()

    set_seed(args.seed)
    use_legit = not args.no_legitimacy
    logger.info(
        f"Training AdaMule | Profile: {args.profile} | Legitimacy Module: {use_legit} | Robust Training: {args.robust}"
    )

    dataset = TransactionGraphDataset(profile=args.profile, seed=args.seed)
    graph = dataset.process()

    in_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]

    model_cfg = load_config("configs/model.yaml").get("adamule", {})
    hidden_dim = model_cfg.get("hidden_dim", 64)
    loss_weights = model_cfg.get("loss_weights", {})

    model = AdaMule(
        in_dim=in_dim,
        edge_dim=edge_dim,
        hidden_dim=hidden_dim,
        use_legitimacy_module=use_legit,
        lambda_fraud=loss_weights.get("lambda_fraud", 1.0),
        lambda_legitimacy=loss_weights.get("lambda_legitimacy", 0.5),
        lambda_robustness=loss_weights.get("lambda_robustness", 0.3)
    )

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-4)
    out_dir = get_project_root() / "outputs" / "models" / "adamule"

    if args.robust:
        robust_trainer = AdversarialTrainer(
            model=model,
            optimizer=optimizer,
            attack_method=args.attack_method,
            output_dir=out_dir,
            device=args.device,
            seed=args.seed
        )
        results = robust_trainer.train_min_max(
            graph=graph,
            rounds=args.rounds,
            epochs_per_round=max(5, args.epochs // args.rounds)
        )
        logger.info("Completed AdaMule Robust Adversarial Training!")
    else:
        trainer = GNNTrainer(
            model=model,
            optimizer=optimizer,
            device=args.device,
            patience=10,
            output_dir=out_dir
        )
        results = trainer.fit(graph=graph, epochs=args.epochs)
        logger.info(f"Completed AdaMule standard training! Best Val F1: {results['best_val_f1']:.4f}")


if __name__ == "__main__":
    main()
