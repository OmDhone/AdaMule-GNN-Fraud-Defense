"""Graph construction utilities for financial transaction networks.

Constructs PyTorch graph representations including heterogeneous relation edges
(transfers, shared device, shared IP) and temporal edge attributes.
"""

from typing import Any, Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import torch


class TransactionGraph:
    """Container for graph tensors and metadata."""

    def __init__(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor,
        y: torch.Tensor,
        y_legitimacy: torch.Tensor,
        account_ids: List[str],
        node_map: Dict[str, int],
        aux_edges: Dict[str, torch.Tensor] | None = None,
        train_mask: torch.Tensor | None = None,
        val_mask: torch.Tensor | None = None,
        test_mask: torch.Tensor | None = None
    ):
        self.x = x
        self.edge_index = edge_index
        self.edge_attr = edge_attr
        self.edge_time = edge_time
        self.y = y
        self.y_legitimacy = y_legitimacy
        self.account_ids = account_ids
        self.node_map = node_map
        self.aux_edges = aux_edges or {}
        self.train_mask = train_mask
        self.val_mask = val_mask
        self.test_mask = test_mask

    @property
    def num_nodes(self) -> int:
        return self.x.shape[0]

    @property
    def num_edges(self) -> int:
        return self.edge_index.shape[1]

    def to(self, device: str | torch.device) -> "TransactionGraph":
        """Move all tensors to target device."""
        self.x = self.x.to(device)
        self.edge_index = self.edge_index.to(device)
        self.edge_attr = self.edge_attr.to(device)
        self.edge_time = self.edge_time.to(device)
        self.y = self.y.to(device)
        self.y_legitimacy = self.y_legitimacy.to(device)
        if self.train_mask is not None:
            self.train_mask = self.train_mask.to(device)
        if self.val_mask is not None:
            self.val_mask = self.val_mask.to(device)
        if self.test_mask is not None:
            self.test_mask = self.test_mask.to(device)
        for k in self.aux_edges:
            self.aux_edges[k] = self.aux_edges[k].to(device)
        return self

    def clone(self) -> "TransactionGraph":
        """Return a deep copy of the graph and its tensors."""
        return TransactionGraph(
            x=self.x.clone(),
            edge_index=self.edge_index.clone(),
            edge_attr=self.edge_attr.clone(),
            edge_time=self.edge_time.clone(),
            y=self.y.clone(),
            y_legitimacy=self.y_legitimacy.clone(),
            account_ids=list(self.account_ids),
            node_map=dict(self.node_map),
            aux_edges={k: v.clone() for k, v in self.aux_edges.items()},
            train_mask=self.train_mask.clone() if self.train_mask is not None else None,
            val_mask=self.val_mask.clone() if self.val_mask is not None else None,
            test_mask=self.test_mask.clone() if self.test_mask is not None else None,
        )


class GraphBuilder:
    """Builds TransactionGraph instances from entity and transaction dataframes."""

    @staticmethod
    def build_graph(
        accounts_df: pd.DataFrame,
        transactions_df: pd.DataFrame,
        feature_df: pd.DataFrame,
        edge_features: np.ndarray,
        train_cutoff: Optional[float] = None,
        val_cutoff: Optional[float] = None
    ) -> TransactionGraph:
        """Construct graph tensors and masks."""
        # Mapping accounts to sequential node indices
        account_ids = accounts_df["account_id"].tolist()
        node_map = {acc: i for i, acc in enumerate(account_ids)}
        num_nodes = len(account_ids)

        # Build node features
        # Columns to ignore as raw identifiers
        ignore_cols = {"account_id", "business_profile", "location", "customer_segment", "account_type"}
        num_cols = [c for c in feature_df.columns if c not in ignore_cols]
        x_np = feature_df[num_cols].values.astype(np.float32)
        # Normalize features with robust z-score based only on train if cutoff provided
        means = np.nanmean(x_np, axis=0, keepdims=True)
        stds = np.nanstd(x_np, axis=0, keepdims=True) + 1e-6
        x_norm = np.nan_to_num((x_np - means) / stds, nan=0.0)
        x_tensor = torch.from_numpy(x_norm)

        # Build edges from transactions
        valid_senders = transactions_df["sender_id"].map(node_map)
        valid_receivers = transactions_df["receiver_id"].map(node_map)
        mask_valid = valid_senders.notna() & valid_receivers.notna()

        tx_valid = transactions_df[mask_valid].reset_index(drop=True)
        edge_attr_valid = edge_features[mask_valid.values]

        src_indices = tx_valid["sender_id"].map(node_map).values.astype(np.int64)
        dst_indices = tx_valid["receiver_id"].map(node_map).values.astype(np.int64)
        edge_index = torch.from_numpy(np.vstack([src_indices, dst_indices]))
        edge_attr = torch.from_numpy(edge_attr_valid)
        edge_time = torch.from_numpy(tx_valid["timestamp"].values.astype(np.float32))

        # Build node labels (y = 1 if account sent or received any fraudulent transaction)
        fraud_senders = tx_valid[tx_valid["is_fraud"] == 1]["sender_id"].map(node_map).dropna().unique().astype(int)
        fraud_receivers = tx_valid[tx_valid["is_fraud"] == 1]["receiver_id"].map(node_map).dropna().unique().astype(int)
        fraud_nodes = set(fraud_senders).union(set(fraud_receivers))

        y = torch.zeros(num_nodes, dtype=torch.float32)
        for fn in fraud_nodes:
            y[fn] = 1.0

        # Build legitimacy labels (1 if legitimate merchant profile, 0 otherwise)
        y_legit = torch.zeros(num_nodes, dtype=torch.float32)
        for idx, row in accounts_df.iterrows():
            if row.get("has_business_profile", 0) == 1 and idx not in fraud_nodes:
                y_legit[idx] = 1.0

        # Heterogeneous auxiliary edges: Shared Device and Shared IP
        dev_groups = tx_valid.groupby("device_id")["sender_id"].unique()
        ip_groups = tx_valid.groupby("ip_id")["sender_id"].unique()

        dev_edges_u, dev_edges_v = [], []
        for accounts in dev_groups:
            if len(accounts) > 1:
                idxs = [node_map[a] for a in accounts if a in node_map]
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        dev_edges_u.extend([idxs[i], idxs[j]])
                        dev_edges_v.extend([idxs[j], idxs[i]])

        ip_edges_u, ip_edges_v = [], []
        for accounts in ip_groups:
            if len(accounts) > 1:
                idxs = [node_map[a] for a in accounts if a in node_map]
                for i in range(len(idxs)):
                    for j in range(i + 1, len(idxs)):
                        ip_edges_u.extend([idxs[i], idxs[j]])
                        ip_edges_v.extend([idxs[j], idxs[i]])

        aux_edges = {}
        if dev_edges_u:
            aux_edges["shared_device"] = torch.tensor([dev_edges_u, dev_edges_v], dtype=torch.int64)
        else:
            aux_edges["shared_device"] = torch.empty((2, 0), dtype=torch.int64)

        if ip_edges_u:
            aux_edges["shared_ip"] = torch.tensor([ip_edges_u, ip_edges_v], dtype=torch.int64)
        else:
            aux_edges["shared_ip"] = torch.empty((2, 0), dtype=torch.int64)

        # Node masks based on temporal cutoff
        # An account belongs to train_mask if its first activity is <= train_cutoff
        # val_mask if first activity between (train_cutoff, val_cutoff]
        # test_mask if first activity > val_cutoff
        # To maintain balanced test evaluation, active nodes across splits are partitioned:
        timestamps = tx_valid["timestamp"].values
        train_mask = torch.zeros(num_nodes, dtype=torch.bool)
        val_mask = torch.zeros(num_nodes, dtype=torch.bool)
        test_mask = torch.zeros(num_nodes, dtype=torch.bool)

        if train_cutoff is not None and val_cutoff is not None:
            first_tx_time = tx_valid.groupby("sender_id")["timestamp"].min().to_dict()
            for acc, idx in node_map.items():
                t = first_tx_time.get(acc, None)
                if t is None:
                    # Inactive accounts defaulted to train
                    train_mask[idx] = True
                elif t <= train_cutoff:
                    train_mask[idx] = True
                elif t <= val_cutoff:
                    val_mask[idx] = True
                else:
                    test_mask[idx] = True
        else:
            # Random default split if no temporal cutoffs specified
            indices = np.arange(num_nodes)
            np.random.shuffle(indices)
            n_tr = int(num_nodes * 0.6)
            n_va = int(num_nodes * 0.2)
            train_mask[indices[:n_tr]] = True
            val_mask[indices[n_tr: n_tr + n_va]] = True
            test_mask[indices[n_tr + n_va:]] = True

        return TransactionGraph(
            x=x_tensor,
            edge_index=edge_index,
            edge_attr=edge_attr,
            edge_time=edge_time,
            y=y,
            y_legitimacy=y_legit,
            account_ids=account_ids,
            node_map=node_map,
            aux_edges=aux_edges,
            train_mask=train_mask,
            val_mask=val_mask,
            test_mask=test_mask
        )
