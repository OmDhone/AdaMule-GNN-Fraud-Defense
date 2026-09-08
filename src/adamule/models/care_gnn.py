"""CARE-GNN (Camouflage-Resistant Graph Neural Network) baseline approximation.

NOTE: Implementation Type: Reproduced and simplified relation-aware neighborhood
aggregation with neighbor similarity filtering (CARE-GNN approximation).
Not an official reproduction.

Reference:
    Dou et al., "Enhancing Graph Neural Network-based Fraud Detectors against
    Camouflaged Fraudsters", CIKM 2020.
"""

from typing import Dict, List, Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class RelationAggregator(nn.Module):
    """Aggregates neighbor features along a specific heterogeneous edge relation with similarity gating."""

    def __init__(self, in_dim: int, out_dim: int, dropout: float = 0.2):
        super().__init__()
        self.linear = nn.Linear(in_dim, out_dim)
        self.sim_weight = nn.Parameter(torch.empty(in_dim, in_dim))
        self.dropout = dropout
        nn.init.xavier_uniform_(self.sim_weight)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, filter_ratio: float = 0.7) -> torch.Tensor:
        if edge_index.size(1) == 0:
            return torch.zeros((x.size(0), self.linear.out_features), device=x.device)

        src, dst = edge_index[0], edge_index[1]
        N = x.size(0)

        # Compute cosine similarity / bilateral similarity score between connected nodes
        h_src = x[src]
        h_dst = x[dst]
        sim_scores = torch.cosine_similarity(torch.matmul(h_src, self.sim_weight), h_dst, dim=-1)
        sim_scores = (sim_scores + 1.0) / 2.0  # scale to [0, 1]

        # Top-p filtering: soft gating
        gate = torch.sigmoid((sim_scores - (1.0 - filter_ratio)) * 5.0)

        # Transformed features
        h_trans = self.linear(h_src) * gate.unsqueeze(-1)

        # Scatter sum
        out = torch.zeros((N, self.linear.out_features), device=x.device)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(h_trans), h_trans)

        deg = torch.zeros(N, device=x.device)
        deg.scatter_add_(0, dst, gate)
        deg = deg.clamp(min=1.0).unsqueeze(-1)
        return out / deg


class CAREGNN(nn.Module):
    """CARE-GNN relation-aware model approximation."""

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
        self.num_relations = len(self.relation_names)
        self.dropout = dropout

        # Feature encoder
        self.feat_encoder = nn.Linear(in_dim, hidden_dim)

        # Relation aggregators
        self.relation_aggs = nn.ModuleDict({
            rel: RelationAggregator(hidden_dim, hidden_dim, dropout=dropout)
            for rel in self.relation_names
        })

        # Inter-relation attention
        self.rel_att = nn.Linear(hidden_dim, 1)

        # Final classification
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_dim, out_dim)
        )

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        aux_edges: Optional[Dict[str, torch.Tensor]] = None
    ) -> torch.Tensor:
        h0 = F.relu(self.feat_encoder(x))
        aux = aux_edges or {}

        # Aggregate along each relation
        relation_embeds = []
        for rel in self.relation_names:
            if rel == "transfers":
                e_idx = edge_index
            else:
                e_idx = aux.get(rel, torch.empty((2, 0), dtype=torch.long, device=x.device))
            h_rel = self.relation_aggs[rel](h0, e_idx)
            relation_embeds.append(h_rel)

        # Stack [num_relations, N, hidden_dim]
        stacked = torch.stack(relation_embeds, dim=0)

        # Attention weights across relations
        att_scores = self.rel_att(stacked).squeeze(-1)  # [num_relations, N]
        att_weights = F.softmax(att_scores, dim=0).unsqueeze(-1)  # [num_relations, N, 1]

        # Multi-relation aggregated representation
        h_neighbors = (stacked * att_weights).sum(dim=0)  # [N, hidden_dim]

        # Combine self-embedding and neighbor embedding
        h_combined = torch.cat([h0, h_neighbors], dim=-1)
        logits = self.classifier(h_combined).squeeze(-1)
        return logits
