#!/usr/bin/env python3
"""CLI script to generate synthetic transaction data and entity tables.

Usage:
    python scripts/generate_data.py --profile development --seed 42
"""

import argparse
import sys
from pathlib import Path

# Ensure src is on path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from adamule.data.generator import SyntheticDataGenerator
from adamule.utils.config import get_profile_config, get_project_root
from adamule.utils.logging import get_logger

logger = get_logger("scripts.generate_data")


def main():
    parser = argparse.ArgumentParser(
        description="Generate synthetic transaction graph and entity data for AdaMule.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--profile",
        type=str,
        default="development",
        choices=["development", "medium", "research"],
        help="Configuration profile for dataset scaling."
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for deterministic generation."
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Custom output directory for raw CSV files (defaults to data/raw)."
    )
    args = parser.parse_args()

    cfg = get_profile_config(args.profile)
    logger.info(f"Loaded configuration for profile: {args.profile}")
    logger.info(f"Target Accounts: {cfg['num_accounts']}, Transactions: {cfg['num_transactions']}")

    generator = SyntheticDataGenerator(config=cfg, seed=args.seed)
    out_dir = args.output_dir or (get_project_root() / "data" / "raw")
    generator.save(output_dir=out_dir)
    logger.info(f"Data generation completed successfully. Files written to: {out_dir}")


if __name__ == "__main__":
    main()
