"""Test suite verifying all baseline and primary GNN model architectures."""

import pytest
import torch
from adamule.models.gcn import GCN
from adamule.models.gat import GAT
from adamule.models.care_gnn import CAREGNN
from adamule.models.fraudre import FRAUDRE
from adamule.models.temporal import TemporalGNN
from adamule.models.adamule import AdaMule


@pytest.fixture
def dummy_graph_data():
    """Create lightweight synthetic graph tensors for fast model testing."""
    N = 50
    in_dim = 16
    edge_dim = 12
    E = 120

    x = torch.randn(N, in_dim)
    src = torch.randint(0, N, (E,))
    dst = torch.randint(0, N, (E,))
    edge_index = torch.stack([src, dst], dim=0)
    edge_attr = torch.rand(E, edge_dim)
    edge_time = torch.linspace(1000.0, 5000.0, E)
    y = torch.randint(0, 2, (N,)).float()
    y_legit = torch.randint(0, 2, (N,)).float()
    mask = torch.rand(N) > 0.3

    aux_edges = {
        "transfers": edge_index,
        "shared_device": torch.randint(0, N, (2, 40)),
        "shared_ip": torch.randint(0, N, (2, 30))
    }

    return {
        "x": x,
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "edge_time": edge_time,
        "y": y,
        "y_legit": y_legit,
        "mask": mask,
        "aux_edges": aux_edges,
        "N": N,
        "in_dim": in_dim,
        "edge_dim": edge_dim
    }


def test_gcn_forward_and_backward(dummy_graph_data):
    d = dummy_graph_data
    model = GCN(in_dim=d["in_dim"], hidden_dim=32, num_layers=2)
    logits = model(d["x"], d["edge_index"])
    
    assert logits.shape == (d["N"],)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[d["mask"]], d["y"][d["mask"]])
    loss.backward()
    
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_gat_forward_and_backward(dummy_graph_data):
    d = dummy_graph_data
    model = GAT(in_dim=d["in_dim"], hidden_dim=32, num_heads=4, num_layers=2)
    logits = model(d["x"], d["edge_index"])
    
    assert logits.shape == (d["N"],)
    loss = torch.nn.functional.binary_cross_entropy_with_logits(logits[d["mask"]], d["y"][d["mask"]])
    loss.backward()
    
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None


def test_care_gnn_forward(dummy_graph_data):
    d = dummy_graph_data
    model = CAREGNN(in_dim=d["in_dim"], hidden_dim=32)
    logits = model(d["x"], d["edge_index"], aux_edges=d["aux_edges"])
    
    assert logits.shape == (d["N"],)
    assert not torch.isnan(logits).any()


def test_fraudre_forward(dummy_graph_data):
    d = dummy_graph_data
    model = FRAUDRE(in_dim=d["in_dim"], hidden_dim=32)
    logits = model(d["x"], d["edge_index"], aux_edges=d["aux_edges"])
    
    assert logits.shape == (d["N"],)
    assert not torch.isnan(logits).any()


def test_temporal_gnn_forward(dummy_graph_data):
    d = dummy_graph_data
    model = TemporalGNN(in_dim=d["in_dim"], edge_dim=d["edge_dim"], hidden_dim=32)
    logits = model(d["x"], d["edge_index"], d["edge_attr"], d["edge_time"])
    
    assert logits.shape == (d["N"],)
    assert not torch.isnan(logits).any()


def test_adamule_forward_and_loss(dummy_graph_data):
    d = dummy_graph_data
    model = AdaMule(
        in_dim=d["in_dim"],
        edge_dim=d["edge_dim"],
        hidden_dim=32,
        use_legitimacy_module=True
    )
    logits = model(d["x"], d["edge_index"], d["edge_attr"], d["edge_time"])
    embeds = model.encode(d["x"], d["edge_index"], d["edge_attr"], d["edge_time"])
    
    assert logits.shape == (d["N"],)
    assert embeds.shape == (d["N"], 64)
    
    total_loss, loss_dict = model.compute_loss(
        logits=logits,
        y_fraud=d["y"],
        embeddings=embeds,
        y_legitimacy=d["y_legit"],
        mask=d["mask"]
    )
    
    assert "loss_fraud" in loss_dict
    assert "loss_legitimacy" in loss_dict
    assert "loss_total" in loss_dict
    assert total_loss.item() > 0.0
    
    total_loss.backward()
    for p in model.parameters():
        if p.requires_grad:
            assert p.grad is not None
