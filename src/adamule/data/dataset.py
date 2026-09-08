"""Transaction graph dataset loader and temporal partitioner."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import pandas as pd
import torch

from adamule.data.features import TemporalFeatureExtractor
from adamule.data.generator import SyntheticDataGenerator
from adamule.data.graph_builder import GraphBuilder, TransactionGraph
from adamule.utils.config import get_profile_config, get_project_root
from adamule.utils.logging import get_logger

logger = get_logger("adamule.data.dataset")


class TransactionGraphDataset:
    """Dataset manager for loading, splitting, and preparing transaction graphs."""

    def __init__(self, profile: str = "development", data_dir: Optional[str | Path] = None, seed: int = 42):
        self.profile = profile
        self.config = get_profile_config(profile)
        self.root_dir = Path(data_dir) if data_dir else get_project_root() / "data"
        self.raw_dir = self.root_dir / "raw"
        self.processed_dir = self.root_dir / "processed"
        self.seed = seed

    def load_or_generate_raw(self) -> Tuple[pd.DataFrame, Dict[str, pd.DataFrame]]:
        """Load raw CSV files if present; otherwise generate them."""
        tx_path = self.raw_dir / "transactions.csv"
        acc_path = self.raw_dir / "accounts.csv"
        merch_path = self.raw_dir / "merchants.csv"

        if not (tx_path.exists() and acc_path.exists() and merch_path.exists()):
            logger.info("Raw data not found. Triggering synthetic generator...")
            generator = SyntheticDataGenerator(config=self.config, seed=self.seed)
            self.raw_dir.mkdir(parents=True, exist_ok=True)
            generator.save(self.raw_dir)

        df_txs = pd.read_csv(tx_path)
        entity_dfs = {
            "accounts": pd.read_csv(acc_path),
            "merchants": pd.read_csv(merch_path),
            "devices": pd.read_csv(self.raw_dir / "devices.csv") if (self.raw_dir / "devices.csv").exists() else pd.DataFrame(),
            "ips": pd.read_csv(self.raw_dir / "ips.csv") if (self.raw_dir / "ips.csv").exists() else pd.DataFrame(),
        }
        return df_txs, entity_dfs

    def process(self, force_recompute: bool = False) -> TransactionGraph:
        """Process raw transaction data into a normalized TransactionGraph with temporal splits."""
        self.processed_dir.mkdir(parents=True, exist_ok=True)
        cache_path = self.processed_dir / f"graph_{self.profile}.pt"

        if cache_path.exists() and not force_recompute:
            logger.info(f"Loading cached graph dataset from: {cache_path}")
            cached_data = torch.load(cache_path, weights_only=False)
            return cached_data

        logger.info(f"Processing raw data for profile '{self.profile}'...")
        df_txs, entity_dfs = self.load_or_generate_raw()
        accounts_df = entity_dfs["accounts"]

        # Determine temporal split cutoffs
        temporal_cfg = self.config.get("temporal_split", {"train_ratio": 0.6, "val_ratio": 0.2, "test_ratio": 0.2})
        r_tr = temporal_cfg.get("train_ratio", 0.6)
        r_va = temporal_cfg.get("val_ratio", 0.2)

        min_ts = df_txs["timestamp"].min()
        max_ts = df_txs["timestamp"].max()
        time_span = max_ts - min_ts

        train_cutoff = min_ts + time_span * r_tr
        val_cutoff = min_ts + time_span * (r_tr + r_va)

        logger.info(f"Temporal Split cutoffs: Train <= {train_cutoff:.1f}, Val <= {val_cutoff:.1f}, Test <= {max_ts:.1f}")

        # Feature Extraction
        extractor = TemporalFeatureExtractor()
        feature_df = extractor.compute_account_features(accounts_df, df_txs, cutoff_timestamp=train_cutoff)
        edge_features = extractor.compute_edge_features(df_txs)

        # Graph Building
        graph = GraphBuilder.build_graph(
            accounts_df=accounts_df,
            transactions_df=df_txs,
            feature_df=feature_df,
            edge_features=edge_features,
            train_cutoff=train_cutoff,
            val_cutoff=val_cutoff
        )

        logger.info(
            f"Graph constructed: Nodes={graph.num_nodes}, Edges={graph.num_edges}, "
            f"Train nodes={graph.train_mask.sum().item()}, Val nodes={graph.val_mask.sum().item()}, "
            f"Test nodes={graph.test_mask.sum().item()}, Fraud nodes={graph.y.sum().item()}"
        )

        torch.save(graph, cache_path)
        logger.info(f"Saved processed graph to: {cache_path}")
        return graph
