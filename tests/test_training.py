"""Test suite verifying training pipeline and adversarial min-max loop."""

import pytest
import torch

from adamule.data.dataset import TransactionGraphDataset
from adamule.models.gcn import GCN
from adamule.models.adamule import AdaMule
from adamule.training.trainer import GNNTrainer
from adamule.training.adversarial_trainer import AdversarialTrainer
from adamule.training.early_stopping import EarlyStopping


def test_early_stopping():
    model = torch.nn.Linear(5, 1)
    es = EarlyStopping(patience=2, mode="max")
    
    assert es(0.5, model) is True
    assert es(0.6, model) is True
    assert es(0.55, model) is False
    assert es.early_stop is False
    assert es(0.54, model) is False
    assert es.early_stop is True


def test_gnn_trainer_epoch(tmp_path):
    dataset = TransactionGraphDataset(profile="development")
    graph = dataset.process()
    
    model = GCN(in_dim=graph.x.shape[1], hidden_dim=16)
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    trainer = GNNTrainer(model=model, optimizer=opt, output_dir=tmp_path)
    train_res = trainer.train_epoch(graph)
    assert "loss_total" in train_res
    assert train_res["loss_total"] > 0.0
    
    val_res = trainer.evaluate(graph, mask=graph.val_mask)
    assert "f1" in val_res
    assert "recall" in val_res


def test_adversarial_trainer_round(tmp_path):
    dataset = TransactionGraphDataset(profile="development")
    graph = dataset.process()
    
    model = AdaMule(
        in_dim=graph.x.shape[1],
        edge_dim=graph.edge_attr.shape[1],
        hidden_dim=16,
        use_legitimacy_module=True
    )
    opt = torch.optim.Adam(model.parameters(), lr=0.01)
    
    adv_trainer = AdversarialTrainer(
        model=model,
        optimizer=opt,
        attack_method="heuristic",
        output_dir=tmp_path
    )
    res = adv_trainer.train_min_max(graph=graph, rounds=1, epochs_per_round=2)
    assert "round_history" in res
    assert len(res["round_history"]) == 1
