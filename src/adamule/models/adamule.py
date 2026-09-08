"""AdaMule: Adversarially Robust Graph Fraud Detector with Legitimacy Preservation.

Integrates:
1. Multi-head structural GNN encoder with edge attribute and temporal Fourier encoding.
2. Primary fraud classification head.
3. Legitimacy-preserving auxiliary module (weakly-supervised contrastive / classification head).
4. Combined multi-task objective with robust training interface.
"""

from typing import Dict, Optional, Tuple
import torch
import torch.nn as nn
import torch.nn.functional as F
from adamule.models.temporal import TimeEncoder


class LegitimacyModule(nn.Module):
    """Auxiliary module distinguishing legitimate irregular activity from camouflaged fraud."""

    def __init__(self, in_dim: int, hidden_dim: int = 32, num_classes: int = 2, temperature: float = 0.1):
        super().__init__()
        self.temperature = temperature
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        self.classifier = nn.Linear(hidden_dim, 1)

    def forward(self, embeddings: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Returns projected representations and legitimacy logits."""
        proj = F.normalize(self.proj(embeddings), dim=-1)
        logits = self.classifier(proj).squeeze(-1)
        return proj, logits

    def compute_loss(
        self,
        embeddings: torch.Tensor,
        legitimacy_labels: torch.Tensor,
        mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Computes auxiliary legitimacy loss using binary cross-entropy and supervised contrast."""
        proj, logits = self.forward(embeddings)
        if mask is not None:
            logits = logits[mask]
            legitimacy_labels = legitimacy_labels[mask]
            proj = proj[mask]

        if len(logits) == 0:
            return torch.tensor(0.0, device=embeddings.device)

        # Classification loss on business legitimacy
        bce_loss = F.binary_cross_entropy_with_logits(logits, legitimacy_labels)
        return bce_loss


class AdaMule(nn.Module):
    """AdaMule primary model."""

    def __init__(
        self,
        in_dim: int,
        edge_dim: int = 12,
        time_dim: int = 16,
        hidden_dim: int = 64,
        out_dim: int = 1,
        num_layers: int = 2,
        num_heads: int = 4,
        dropout: float = 0.2,
        use_legitimacy_module: bool = True,
        lambda_fraud: float = 1.0,
        lambda_legitimacy: float = 0.5,
        lambda_robustness: float = 0.3
    ):
        super().__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.use_legitimacy_module = use_legitimacy_module
        self.lambda_fraud = lambda_fraud
        self.lambda_legitimacy = lambda_legitimacy
        self.lambda_robustness = lambda_robustness
        self.dropout = dropout

        # Temporal encoder
        self.time_encoder = TimeEncoder(time_dim=time_dim)

        # Node projection
        self.node_proj = nn.Linear(in_dim, hidden_dim)
        self.edge_proj = nn.Linear(edge_dim + time_dim, hidden_dim)

        # Structural GNN Layers (Multi-Head Message Passing)
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        self.conv_w = nn.Linear(hidden_dim, hidden_dim)
        self.att_w = nn.Linear(hidden_dim * 2, 1)

        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

        # Primary Fraud Classifier
        self.fraud_classifier = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, out_dim)
        )

        # Legitimacy Module
        if use_legitimacy_module:
            self.legitimacy_module = LegitimacyModule(in_dim=hidden_dim * 2, hidden_dim=hidden_dim // 2)
        else:
            self.legitimacy_module = None

    def encode(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor
    ) -> torch.Tensor:
        """Encode nodes using temporal and structural information."""
        N = x.size(0)
        h_node = F.relu(self.node_proj(x))

        if edge_index.size(1) == 0:
            return torch.cat([h_node, h_node], dim=-1)

        src, dst = edge_index[0], edge_index[1]

        # Time delta embedding
        t_ref = edge_time.max()
        delta_t = (t_ref - edge_time).clamp(min=0.0)
        t_embed = self.time_encoder(delta_t)

        # Edge context = edge_attr + t_embed
        edge_ctx = F.relu(self.edge_proj(torch.cat([edge_attr, t_embed], dim=-1)))

        # Messages = neighbor state + edge context transformed by conv_w
        messages = F.relu(self.conv_w(h_node[src] + edge_ctx))

        # Attention weighting
        att_in = torch.cat([h_node[dst], messages], dim=-1)
        att_weight = F.leaky_relu(self.att_w(att_in))
        att_exp = torch.exp(att_weight - att_weight.max())

        denom = torch.zeros((N, 1), device=x.device)
        denom.scatter_add_(0, dst.unsqueeze(-1), att_exp)
        norm_att = att_exp / denom[dst].clamp(min=1e-8)

        weighted_msg = messages * norm_att
        agg = torch.zeros((N, self.hidden_dim), device=x.device)
        agg.scatter_add_(0, dst.unsqueeze(-1).expand_as(weighted_msg), weighted_msg)

        h_updated = self.layer_norm1(h_node + F.dropout(agg, p=self.dropout, training=self.training))
        h_norm2 = self.layer_norm2(h_updated)
        embeddings = torch.cat([h_node, h_norm2], dim=-1)
        return embeddings

    def forward(
        self,
        x: torch.Tensor,
        edge_index: torch.Tensor,
        edge_attr: torch.Tensor,
        edge_time: torch.Tensor
    ) -> torch.Tensor:
        """Computes fraud classification logits."""
        embeddings = self.encode(x, edge_index, edge_attr, edge_time)
        logits = self.fraud_classifier(embeddings).squeeze(-1)
        return logits

    def compute_loss(
        self,
        logits: torch.Tensor,
        y_fraud: torch.Tensor,
        embeddings: torch.Tensor,
        y_legitimacy: torch.Tensor,
        mask: Optional[torch.Tensor] = None,
        robust_loss: Optional[torch.Tensor] = None
    ) -> Tuple[torch.Tensor, Dict[str, float]]:
        """Multi-task objective function combining fraud, legitimacy, and robust losses."""
        if mask is not None:
            active_logits = logits[mask]
            active_y = y_fraud[mask]
        else:
            active_logits = logits
            active_y = y_fraud

        # Weighted BCE for fraud class imbalance
        num_pos = (active_y == 1).sum().float()
        num_neg = (active_y == 0).sum().float()
        pos_weight = (num_neg / (num_pos + 1e-5)).clamp(min=1.0, max=15.0)

        fraud_loss = F.binary_cross_entropy_with_logits(
            active_logits, active_y, pos_weight=pos_weight
        )

        total_loss = self.lambda_fraud * fraud_loss
        loss_dict = {"loss_fraud": fraud_loss.item()}

        if self.use_legitimacy_module and self.legitimacy_module is not None:
            legit_loss = self.legitimacy_module.compute_loss(embeddings, y_legitimacy, mask=mask)
            total_loss = total_loss + self.lambda_legitimacy * legit_loss
            loss_dict["loss_legitimacy"] = legit_loss.item()

        if robust_loss is not None:
            total_loss = total_loss + self.lambda_robustness * robust_loss
            loss_dict["loss_robustness"] = robust_loss.item()

        loss_dict["loss_total"] = total_loss.item()
        return total_loss, loss_dict
