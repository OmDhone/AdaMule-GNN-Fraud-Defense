"""Action primitives for structuring and camouflaging money mule transaction topologies."""

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple
import torch
import numpy as np


@dataclass
class GraphPerturbationAction:
    """Represents a discrete or parametrized perturbation action."""
    action_type: str  # 'smurf_split', 'add_camouflage', 'insert_intermediary', 'time_shift', 'redistribute'
    target_edge_idx: Optional[int] = None
    target_node_idx: Optional[int] = None
    target_counterparty: Optional[int] = None
    amount_split: Optional[List[float]] = None
    time_offset: float = 0.0
    amount: float = 0.0


class ActionApplier:
    """Applies verified perturbation actions to PyTorch TransactionGraph objects."""

    @staticmethod
    def apply_add_camouflage_edge(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor,
        sender: int,
        merchant_receiver: int,
        amount: float,
        timestamp: float,
        edge_dim: int = 12
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Inject a small camouflage transaction edge to a legitimate merchant."""
        new_edge = torch.tensor([[sender], [merchant_receiver]], dtype=torch.long, device=edge_index.device)
        new_edge_index = torch.cat([edge_index, new_edge], dim=1)

        # Build realistic edge attribute vector
        new_attr = torch.zeros((1, edge_dim), dtype=torch.float32, device=edge_attr.device)
        new_attr[0, 0] = np.log1p(amount)
        # channel = qr_code (idx 5), type = p2m (idx 8)
        if edge_dim >= 9:
            new_attr[0, 5] = 1.0  # qr_code
            new_attr[0, 8] = 1.0  # p2m
        new_edge_attr = torch.cat([edge_attr, new_attr], dim=0)

        new_edge_time = torch.cat([
            edge_time,
            torch.tensor([timestamp], dtype=torch.float32, device=edge_time.device)
        ], dim=0)

        return new_edge_index, new_edge_attr, new_edge_time

    @staticmethod
    def apply_smurf_split(
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor,
        edge_idx_to_split: int,
        split_amounts: List[float],
        time_increments: List[float]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Replace a single large transaction with multiple structured smaller transfers."""
        device = edge_index.device
        src = edge_index[0, edge_idx_to_split].item()
        dst = edge_index[1, edge_idx_to_split].item()
        base_time = edge_time[edge_idx_to_split].item()
        base_attr = edge_attr[edge_idx_to_split].clone()

        # Remove the original edge
        keep_mask = torch.ones(edge_index.size(1), dtype=torch.bool, device=device)
        keep_mask[edge_idx_to_split] = False

        filt_edge_index = edge_index[:, keep_mask]
        filt_edge_attr = edge_attr[keep_mask]
        filt_edge_time = edge_time[keep_mask]

        # Add the structured fragments
        new_edges_u = [src] * len(split_amounts)
        new_edges_v = [dst] * len(split_amounts)
        new_edge_tensor = torch.tensor([new_edges_u, new_edges_v], dtype=torch.long, device=device)

        new_attrs = []
        new_times = []
        for i, (amt, dt) in enumerate(zip(split_amounts, time_increments)):
            attr_i = base_attr.clone()
            attr_i[0] = np.log1p(amt)
            new_attrs.append(attr_i)
            new_times.append(base_time + dt)

        out_edge_index = torch.cat([filt_edge_index, new_edge_tensor], dim=1)
        out_edge_attr = torch.cat([filt_edge_attr, torch.stack(new_attrs, dim=0)], dim=0)
        out_edge_time = torch.cat([filt_edge_time, torch.tensor(new_times, dtype=torch.float32, device=device)], dim=0)

        return out_edge_index, out_edge_attr, out_edge_time
