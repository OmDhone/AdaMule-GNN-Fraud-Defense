#!/usr/bin/env python3
"""CLI script to train the RL structuring attacker against a frozen detector.

Usage:
    python scripts/train_attacker.py --model gcn --profile development --timesteps 1000
"""

import argparse
import sys
from pathlib import Path
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adamule.attacks.rl_attacker import RLStructuringAttacker
from adamule.data.dataset import TransactionGraphDataset
from adamule.models.gcn import GCN
from adamule.models.gat import GAT
from adamule.models.adamule import AdaMule
from adamule.utils.config import get_project_root
from adamule.utils.io import load_checkpoint
from adamule.utils.logging import get_logger
from adamule.utils.seed import set_seed

logger = get_logger("scripts.train_attacker")


def main():
    parser = argparse.ArgumentParser(
        description="Train RL structuring attacker policy against a frozen detector.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--model", type=str, default="gcn", choices=["gcn", "gat", "adamule"], help="Target model.")
    parser.add_argument("--profile", type=str, default="development", choices=["development", "medium", "research"])
    parser.add_argument("--timesteps", type=int, default=1500, help="RL training timesteps.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    args = parser.parse_args()

    set_seed(args.seed)
    logger.info(f"Training RL Attacker against {args.model.upper()} | Profile: {args.profile}")

    dataset = TransactionGraphDataset(profile=args.profile, seed=args.seed)
    graph = dataset.process()

    in_dim = graph.x.shape[1]
    edge_dim = graph.edge_attr.shape[1]

    # Instantiate or load model
    ckpt_path = get_project_root() / "outputs" / "models" / f"baseline_{args.model}" / "best_model.pt"
    if args.model == "gcn":
        detector = GCN(in_dim=in_dim, hidden_dim=64)
    elif args.model == "gat":
        detector = GAT(in_dim=in_dim, hidden_dim=64)
    else:
        detector = AdaMule(in_dim=in_dim, edge_dim=edge_dim, hidden_dim=64)

    if ckpt_path.exists():
        logger.info(f"Loading weights from {ckpt_path}")
        ckpt = load_checkpoint(ckpt_path)
        detector.load_state_dict(ckpt["state_dict"])
    else:
        logger.warning("No checkpoint found; training against freshly initialized detector.")

    out_dir = get_project_root() / "outputs" / "models" / f"rl_attacker_{args.model}"
    rl_agent = RLStructuringAttacker(detector=detector, output_dir=out_dir, seed=args.seed)
    rl_agent.train(graph=graph, total_timesteps=args.timesteps)
    logger.info("RL Attacker training completed!")


if __name__ == "__main__":
    main()
