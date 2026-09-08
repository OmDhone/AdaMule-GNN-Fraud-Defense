"""FRAUDRE baseline approximation.

NOTE: Implementation Type: Reproduced and simplified fraud-aware relation enhancement
(FRAUDRE approximation). Not an official reproduction.

Reference:
    Zhang et al., "FRAUDRE: Fraud Detection with Imbalanced Heterogeneous Graph
    Neural Networks", IJCAI 2021.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class FRAUDRE(nn.Module):
    """FRAUDRE approximation for fraud detection on imbalanced graphs."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        relation_names: Optional[List[str]] = None,
        dropout: float = 0.2
    ):
        super().__init__()
        self.relation_names = relation_names or ["transfers", "shared_device", "shared_ip"]
        self.dropout = dropout

        self.input_layer = nn.Linear(in_dim, hidden_dim)

        self.rel_transforms = nn.ModuleDict({
            rel: nn.Linear(hidden_dim, hidden_dim) for rel in self.relation_names
        })

        self.rel_attention = nn.Parameter(torch.ones(len(self.relation_names)))
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        aux_edges: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        N = x.size(0)
        h0 = F.relu(self.input_layer(x))
        aux = aux_edges or {}

        rel_outputs = []
        rel_weights = F.softmax(self.rel_attention, dim=0)

        for idx, rel in enumerate(self.relation_names):
            if rel == "transfers":
                e_idx = edge_index
            else:
                e_idx = aux.get(rel, torch.empty((2, 0), dtype=torch.long, device=x.device))

            if e_idx.size(1) == 0:
                h_rel = torch.zeros((N, h0.size(1)), device=x.device)
            else:
                src, dst = e_idx[0], e_idx[1]
                # Degree normalization
                deg = torch.zeros(N, device=x.device)
                deg.scatter_add_(0, dst, torch.ones(e_idx.size(1), device=x.device))
                deg_inv = torch.pow(deg.clamp(min=1.0), -1.0).unsqueeze(-1)

                msg = self.rel_transforms[rel](h0[src])
                out = torch.zeros((N, h0.size(1)), device=x.device)
                out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
                h_rel = out * deg_inv

            rel_outputs.append(h_rel * rel_weights[idx])

        # Aggregate across all relations
        h_agg = torch.stack(rel_outputs, dim=0).sum(dim=0)
        h_combined = torch.cat([h0, h_agg], dim=-1)
        logits = self.classifier(h_combined).squeeze(-1)
        return logits
