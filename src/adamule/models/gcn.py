"""Graph Convolutional Network (GCN) baseline for graph fraud detection.

Implements symmetric normalized adjacency convolution:
    H^(l+1) = sigma( D_tilde^(-1/2) A_tilde D_tilde^(-1/2) H^(l) W^(l) )
"""

from typing import Optional
import torch
import torch.nn as nn
import torch.nn.functional as F


class GCNLayer(nn.Module):
    """Single GCN convolution layer using sparse/index message passing."""

    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.empty((in_features, out_features)))
        if bias:
            self.bias = nn.Parameter(torch.empty(out_features))
        else:
            self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Forward pass.
        
        Args:
            x: Node feature matrix of shape [N, in_features]
            edge_index: Edge index tensor of shape [2, E]
        """
        N = x.size(0)
        E = edge_index.size(1)
        
        # Add self-loops: A_tilde = A + I
        loop_index = torch.arange(0, N, dtype=torch.long, device=edge_index.device)
        loop_edge = torch.stack([loop_index, loop_index], dim=0)
        full_edge_index = torch.cat([edge_index, loop_edge], dim=1)
        
        src, dst = full_edge_index[0], full_edge_index[1]
        
        # Compute degrees D_tilde
        deg = torch.zeros(N, device=edge_index.device)
        deg.scatter_add_(0, dst, torch.ones(full_edge_index.size(1), device=edge_index.device))
        deg_inv_sqrt = torch.pow(deg.clamp(min=1.0), -0.5)
        
        # Normalized coefficients: norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        norm = deg_inv_sqrt[src] * deg_inv_sqrt[dst]
        
        # Linear transformation
        h = torch.matmul(x, self.weight)
        
        # Message passing: out_i = sum_{j in N(i)} norm_{j, i} * h_j
        msg = h[src] * norm.unsqueeze(1)
        out = torch.zeros_like(h)
        out.scatter_add_(0, dst.unsqueeze(1).expand_as(msg), msg)
        
        if self.bias is not None:
            out = out + self.bias
        return out


class GCN(nn.Module):
    """Multi-layer Graph Convolutional Network baseline."""

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 2,
        dropout: float = 0.2,
        use_batch_norm: bool = True
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.dropout = dropout
        self.use_batch_norm = use_batch_norm

        self.convs = nn.ModuleList()
        self.norms = nn.ModuleList()

        if num_layers == 1:
            self.convs.append(GCNLayer(in_dim, hidden_dim))
        else:
            self.convs.append(GCNLayer(in_dim, hidden_dim))
            if use_batch_norm:
                self.norms.append(nn.BatchNorm1d(hidden_dim))
            for _ in range(num_layers - 2):
                self.convs.append(GCNLayer(hidden_dim, hidden_dim))
                if use_batch_norm:
                    self.norms.append(nn.BatchNorm1d(hidden_dim))
            self.convs.append(GCNLayer(hidden_dim, hidden_dim))

        self.classifier = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Computes fraud classification logits for all nodes.
        
        Returns:
            Logits tensor of shape [N]
        """
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                if self.use_batch_norm and i < len(self.norms):
                    h = self.norms[i](h)
                h = F.relu(h)
                h = F.dropout(h, p=self.dropout, training=self.training)

        logits = self.classifier(h).squeeze(-1)
        return logits

    def get_embeddings(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        """Extract node embeddings prior to the final classification layer."""
        h = x
        for i, conv in enumerate(self.convs):
            h = conv(h, edge_index)
            if i < len(self.convs) - 1:
                if self.use_batch_norm and i < len(self.norms):
                    h = self.norms[i](h)
                h = F.relu(h)
        return h
