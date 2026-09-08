"""GNN architectures for graph fraud detection."""

from adamule.models.gcn import GCN, GCNLayer
from adamule.models.gat import GAT, GATLayer
from adamule.models.care_gnn import CAREGNN
from adamule.models.fraudre import FRAUDRE
from adamule.models.temporal import TemporalGNN, TimeEncoder
from adamule.models.adamule import AdaMule, LegitimacyModule

__all__ = [
    "GCN",
    "GCNLayer",
    "GAT",
    "GATLayer",
    "CAREGNN",
    "FRAUDRE",
    "TemporalGNN",
    "TimeEncoder",
    "AdaMule",
    "LegitimacyModule",
]
