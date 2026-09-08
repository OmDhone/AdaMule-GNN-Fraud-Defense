"""Graph Attention Network (GAT) baseline for graph fraud detection.

Implements multi-head neighborhood self-attention:
    alpha_{i, j} = softmax_j( LeakyReLU( a^T [Wh_i || Wh_j] ) )
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class GATLayer(nn.Module):
    """Single Multi-Head Graph Attention layer."""

    def __init__(
        self,
        in_features: int,
        out_features: int,
        num_heads: int = 4,
        concat: bool = True,
        dropout: float = 0.2,
        negative_slope: float = 0.2
    ):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.num_heads = num_heads
        self.concat = concat
        self.dropout = dropout
        self.negative_slope = negative_slope

        self.lin = nn.Linear(in_features, num_heads * out_features, bias=False)
        self.att_src = nn.Parameter(torch.empty((1, num_heads, out_features)))
        self.att_dst = nn.Parameter(torch.empty((1, num_heads, out_features)))

        if not concat:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.bias = nn.Parameter(torch.zeros(num_heads * out_features))

        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.lin.weight)
        nn.init.xavier_uniform_(self.att_src)
        nn.init.xavier_uniform_(self.att_dst)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        N = x.size(0)
        # Add self-loops
        loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
        loop_edge = torch.stack([loop_index, loop_index], dim=0)
        full_edge_index = torch.cat([edge_index, loop_edge], dim=1)

        src, dst = full_edge_index[0], full_edge_index[1]

        # [N, num_heads, out_features]
        h = self.lin(x).view(-1, self.num_heads, self.out_features)

        # Attention scores
        alpha_src = (h * self.att_src).sum(dim=-1)  # [N, num_heads]
        alpha_dst = (h * self.att_dst).sum(dim=-1)  # [N, num_heads]

        # Edge attention: a_ij = LeakyReLU(alpha_src[src] + alpha_dst[dst])
        edge_att = alpha_src[src] + alpha_dst[dst]  # [E, num_heads]
        edge_att = F.leaky_relu(edge_att, negative_slope=self.negative_slope)

        # Softmax normalization per target node dst
        # Numerically stable exp with max subtraction per dst
        edge_att_exp = torch.exp(edge_att - edge_att.max())
        denom = torch.zeros(N, self.num_heads, device=edge_index.device)
        denom.scatter_add_(0, dst.unsqueeze(1).expand_as(edge_att_exp), edge_att_exp)
        denom = denom.clamp(min=1e-8)
        norm_att = edge_att_exp / denom[dst]  # [E, num_heads]

        if self.training and self.dropout > 0:
            norm_att = F.dropout(norm_att, p=self.dropout, training=True)

        # Message passing: out_dst = sum_src norm_att * h_src
        msg = h[src] * norm_att.unsqueeze(-1)  # [E, num_heads, out_features]
        out = torch.zeros(N, self.num_heads, self.out_features, device=edge_index.device)
        out.scatter_add_(0, dst.unsqueeze(1).unsqueeze(2).expand_as(msg), msg)

        if self.concat:
            out = out.view(N, self.num_heads * self.out_features)
        else:
            out = out.mean(dim=1)

        out = out + self.bias
        return out


class GAT(nn.Module):
    """Multi-layer Graph Attention Network baseline."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.num_heads = num_heads
        self.dropout = dropout

        self.convs = nn.ModuleList()
        # Layer 1
        self.convs.append(GATLayer(in_dim, hidden_dim // num_heads, num_heads=num_heads, concat=True, dropout=dropout))
        # Middle layers
        for _ in range(num_layers - 2):
            self.convs.append(GATLayer(hidden_dim, hidden_dim // num_heads, num_heads=num_heads, concat=True, dropout=dropout))
        # Final layer
        self.convs.append(GATLayer(hidden_dim, hidden_dim, num_heads=1, concat=False, dropout=dropout))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.elu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h).squeeze(-1)
        return logits

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                h = F.elu(h)
        return h
