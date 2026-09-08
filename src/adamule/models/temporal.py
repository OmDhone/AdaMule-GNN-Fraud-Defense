"""Temporal Graph Neural Network baseline with continuous Fourier time-encoding.

Implements TGAT-style harmonic time encoding and temporal message aggregation:
    Phi(t) = [cos(w_1 * t), sin(w_1 * t), ..., cos(w_d * t), sin(w_d * t)]
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class TimeEncoder(nn.Module):
    """Fourier sinusoidal time encoder mapping scalar time deltas to continuous vectors."""

    def __init__(self, time_dim: int = 16):
        super().__init__()
        self.time_dim = time_dim
        # Log-spaced frequencies
        factor = 1.0 / (10000.0 ** (torch.arange(0, time_dim, 2).float() / time_dim))
        self.register_buffer("factor", factor)

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        """Args: t of shape [E] or [E, 1]. Returns: [E, time_dim]."""
        if t.dim() == 1:
            t = t.unsqueeze(-1)
        phase = t * self.factor.unsqueeze(0)
        return torch.cat([torch.cos(phase), torch.sin(phase)], dim=-1)


class TemporalGNN(nn.Module):
    """Temporal Graph Network baseline combining time-encoding and edge features."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int = 12,
        time_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 1,
        dropout: float = 0.2
    ):
        super().__init__()
        self.time_encoder = TimeEncoder(time_dim=time_dim)
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        
        # Message dimension = node_dim + edge_dim + time_dim
        msg_dim = hidden_dim + edge_dim + time_dim
        self.msg_linear = nn.Linear(msg_dim, hidden_dim)

        self.temporal_att = nn.Linear(hidden_dim, 1)
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
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor
    ) -> torch.Tensor:
        N = x.size(0)
        E = edge_index.size(1)
        h_node = F.relu(self.node_proj(x))

        if E == 0:
            h_combined = torch.cat([h_node, torch.zeros_like(h_node)], dim=-1)
            return self.classifier(h_combined).squeeze(-1)

        src, dst = edge_index[0], edge_index[1]

        # Compute relative timestamps (difference from max timestamp)
        t_ref = edge_time.max()
        delta_t = (t_ref - edge_time).clamp(min=0.0)
        t_embed = self.time_encoder(delta_t)

        # Message formulation: [h_src, edge_attr, t_embed]
        raw_msg = torch.cat([h_node[src], edge_attr, t_embed], dim=-1)
        msg = F.relu(self.msg_linear(raw_msg))

        # Temporal attention
        att_score = F.leaky_relu(self.temporal_att(msg))
        att_exp = torch.exp(att_score - att_score.max())
        denom = torch.zeros((N, 1), device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(-1), att_exp)
        denom = denom.clamp(min=1e-8)
        norm_att = att_exp / denom[dst]

        # Aggregate weighted messages to dst nodes
        weighted_msg = msg * norm_att
        agg_msg = torch.zeros((N, h_node.size(1)), device=x.device)
        agg_msg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_msg), weighted_msg)

        h_final = torch.cat([h_node, agg_msg], dim=-1)
        logits = self.classifier(h_final).squeeze(-1)
        return logits
