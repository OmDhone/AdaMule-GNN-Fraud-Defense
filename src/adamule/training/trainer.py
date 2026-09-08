"""Unified trainer for baseline GNNs and AdaMule."""

from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import torch
import torch.nn as nn
from tqdm import tqdm

from adamule.data.graph_builder import TransactionGraph
from adamule.evaluation.metrics import compute_classification_metrics
from adamule.training.early_stopping import EarlyStopping
from adamule.training.losses import compute_weighted_bce
from adamule.utils.io import save_checkpoint, save_json
from adamule.utils.logging import get_logger

logger = get_logger("adamule.training.trainer")


class GNNTrainer:
    """Trains and evaluates GNN models for graph fraud detection."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        device: str = "cpu",
        patience: int = 10,
        output_dir: Optional[str | Path] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/models")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.early_stopping = EarlyStopping(patience=patience, mode="max")

    def _forward_model(self, graph: TransactionGraph) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
        """Call model with signature matching its class type."""
        m_name = self.model.__class__.__name__.lower()
        if "adamule" in m_name:
            logits = self.model(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
            embeddings = self.model.encode(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
            return logits, embeddings
        elif "temporal" in m_name:
            logits = self.model(graph.x, graph.edge_index, graph.edge_attr, graph.edge_time)
            return logits, None
        elif "caregnn" in m_name or "fraudre" in m_name:
            logits = self.model(graph.x, graph.edge_index, aux_edges=graph.aux_edges)
            return logits, None
        else:
            # GCN, GAT
            logits = self.model(graph.x, graph.edge_index)
            return logits, None

    def train_epoch(self, graph: TransactionGraph) -> Dict[str, float]:
        """Run a single training epoch."""
        self.model.train()
        self.optimizer.zero_grad()

        graph = graph.to(self.device)
        logits, embeddings = self._forward_model(graph)

        m_name = self.model.__class__.__name__.lower()
        if "adamule" in m_name:
            loss, loss_dict = self.model.compute_loss(
                logits=logits,
                y_fraud=graph.y,
                embeddings=embeddings,
                y_legitimacy=graph.y_legitimacy,
                mask=graph.train_mask
            )
        else:
            loss = compute_weighted_bce(logits[graph.train_mask], graph.y[graph.train_mask])
            loss_dict = {"loss_total": loss.item()}

        loss.backward()
        self.optimizer.step()
        return loss_dict

    @torch.no_grad()
    def evaluate(
        self,
        graph: TransactionGraph,
        mask: Optional[torch.Tensor] = None
    ) -> Dict[str, float]:
        """Evaluate model on specified split mask (or test mask by default)."""
        self.model.eval()
        graph = graph.to(self.device)
        logits, _ = self._forward_model(graph)

        eval_mask = mask if mask is not None else graph.test_mask
        y_true = graph.y[eval_mask]
        y_prob = torch.sigmoid(logits[eval_mask])

        # Mask for legitimate irregular merchants (hard negatives: y_legitimacy == 1 & y == 0)
        hard_neg_mask = (graph.y_legitimacy[eval_mask] == 1) & (y_true == 0)

        metrics = compute_classification_metrics(
            y_true=y_true,
            y_prob=y_prob,
            threshold=0.5,
            hard_negative_mask=hard_neg_mask
        )
        return metrics

    def fit(
        self,
        graph: TransactionGraph,
        epochs: int = 50,
        eval_every: int = 1
    ) -> Dict[str, Any]:
        """Train model until convergence or max epochs with early stopping."""
        history = []
        best_val_f1 = 0.0

        for epoch in range(1, epochs + 1):
            train_losses = self.train_epoch(graph)

            if epoch % eval_every == 0 or epoch == epochs:
                val_metrics = self.evaluate(graph, mask=graph.val_mask)
                val_f1 = val_metrics["f1"]

                is_best = self.early_stopping(val_f1, self.model)
                if is_best:
                    best_val_f1 = val_f1

                record = {
                    "epoch": epoch,
                    **train_losses,
                    "val_f1": val_f1,
                    "val_roc_auc": val_metrics["roc_auc"],
                    "val_pr_auc": val_metrics["pr_auc"],
                    "val_recall": val_metrics["recall"],
                    "val_precision": val_metrics["precision"],
                }
                history.append(record)

                if self.early_stopping.early_stop:
                    logger.info(f"Early stopping triggered at epoch {epoch}")
                    break

        # Load best checkpoint weights
        self.early_stopping.load_best(self.model)

        # Final Test Evaluation
        test_metrics = self.evaluate(graph, mask=graph.test_mask)
        logger.info(
            f"Test Evaluation Results -> Precision: {test_metrics['precision']}, "
            f"Recall: {test_metrics['recall']}, F1: {test_metrics['f1']}, "
            f"ROC-AUC: {test_metrics['roc_auc']}, PR-AUC: {test_metrics['pr_auc']}, "
            f"Hard-Negative FPR: {test_metrics.get('hard_negative_fpr', 0.0)}"
        )

        # Save artifacts
        save_checkpoint(
            self.model.state_dict(),
            self.output_dir / "best_model.pt",
            metadata={"test_metrics": test_metrics, "best_val_f1": best_val_f1}
        )
        save_json(test_metrics, self.output_dir / "test_metrics.json")
        save_json({"history": history}, self.output_dir / "training_history.json")

        return {
            "test_metrics": test_metrics,
            "best_val_f1": best_val_f1,
            "history": history
        }
