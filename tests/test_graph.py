"""Test suite verifying graph construction, feature extraction, and temporal split integrity."""

import pytest
import torch
import numpy as np
from adamule.data.dataset import TransactionGraphDataset
from adamule.data.graph_builder import TransactionGraph


def test_graph_construction_and_tensors():
    """Verify that TransactionGraph correctly initializes and tensor dimensions align."""
    dataset = TransactionGraphDataset(profile="development")
    graph = dataset.process(force_recompute=True)
    
    assert isinstance(graph, TransactionGraph)
    assert graph.num_nodes > 0
    assert graph.num_edges > 0
    
    # Check node feature matrix x
    assert graph.x.dim() == 2
    assert graph.x.shape[0] == graph.num_nodes
    assert not torch.isnan(graph.x).any(), "Node features contain NaNs"
    
    # Check edge_index
    assert graph.edge_index.dim() == 2
    assert graph.edge_index.shape[0] == 2
    assert graph.edge_index.shape[1] == graph.num_edges
    assert graph.edge_index.max() < graph.num_nodes
    assert graph.edge_index.min() >= 0
    
    # Check edge_attr and edge_time
    assert graph.edge_attr.shape[0] == graph.num_edges
    assert graph.edge_time.shape[0] == graph.num_edges
    
    # Check labels
    assert graph.y.shape[0] == graph.num_nodes
    assert graph.y_legitimacy.shape[0] == graph.num_nodes
    assert ((graph.y == 0) | (graph.y == 1)).all()


def test_temporal_split_and_no_leakage():
    """Verify that temporal masks are disjoint, non-empty, and feature extraction prevents leakage."""
    dataset = TransactionGraphDataset(profile="development")
    graph = dataset.process(force_recompute=False)
    
    assert graph.train_mask is not None
    assert graph.val_mask is not None
    assert graph.test_mask is not None
    
    # Check masks are mutually exclusive
    overlap_tr_va = (graph.train_mask & graph.val_mask).sum().item()
    overlap_tr_te = (graph.train_mask & graph.test_mask).sum().item()
    overlap_va_te = (graph.val_mask & graph.test_mask).sum().item()
    assert overlap_tr_va == 0, "Train and Val masks overlap"
    assert overlap_tr_te == 0, "Train and Test masks overlap"
    assert overlap_va_te == 0, "Val and Test masks overlap"
    
    # Check all active nodes belong to splits
    total_assigned = (graph.train_mask | graph.val_mask | graph.test_mask).sum().item()
    assert total_assigned == graph.num_nodes
    
    # Check positive fraud instances exist across splits
    train_fraud = graph.y[graph.train_mask].sum().item()
    test_fraud = graph.y[graph.test_mask].sum().item()
    assert train_fraud > 0, "No fraud nodes in train split"
    assert test_fraud > 0, "No fraud nodes in test split"


def test_auxiliary_heterogeneous_edges():
    """Verify auxiliary relation edges (shared device, shared IP)."""
    dataset = TransactionGraphDataset(profile="development")
    graph = dataset.process(force_recompute=False)
    
    assert "shared_device" in graph.aux_edges
    assert "shared_ip" in graph.aux_edges
    dev_edges = graph.aux_edges["shared_device"]
    assert dev_edges.shape[0] == 2
    if dev_edges.shape[1] > 0:
        assert dev_edges.max() < graph.num_nodes
