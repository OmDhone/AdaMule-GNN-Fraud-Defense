"""Min-Max Alternating Adversarial / Robust Trainer for AdaMule.

Implements the iterative closed-loop defense:
    min_theta max_delta L(theta; G + delta)
    
Alternates between:
1. Training detector
2. Freezing detector and running constrained evasion attacker
3. Harvesting evasive/hardened graph modifications into training replay buffer
4. Unfreezing detector and retraining on hardened examples
"""

from pathlib import Path
from typing import Any, Dict, List, Optional
import torch
import torch.nn as nn

from adamule.attacks.heuristic import HeuristicAttacker
from adamule.attacks.gradient_attack import ConstrainedGradientAttacker
from adamule.data.graph_builder import TransactionGraph
from adamule.evaluation.metrics import compute_classification_metrics, compute_robustness_metrics
from adamule.training.trainer import GNNTrainer
from adamule.utils.io import save_checkpoint, save_json
from adamule.utils.logging import get_logger

logger = get_logger("adamule.training.adversarial_trainer")


class AdversarialTrainer:
    """Orchestrates alternating min-max robust training."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        attack_method: str = "heuristic",
        output_dir: Optional[str | Path] = None,
        device: str = "cpu",
        seed: int = 42
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.attack_method = attack_method
        self.device = device
        self.output_dir = Path(output_dir) if output_dir else Path("outputs/models/robust_adamule")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.seed = seed

    def train_min_max(
        self,
        graph: TransactionGraph,
        rounds: int = 3,
        epochs_per_round: int = 15,
        adversarial_ratio: float = 0.3
    ) -> Dict[str, Any]:
        """Run iterative min-max robust training loop."""
        logger.info(
            f"Starting Adversarial Robust Training: Rounds={rounds}, "
            f"Epochs/round={epochs_per_round}, Attack={self.attack_method}"
        )

        base_trainer = GNNTrainer(
            model=self.model,
            optimizer=self.optimizer,
            device=self.device,
            patience=epochs_per_round,
            output_dir=self.output_dir
        )

        current_graph = graph.clone()
        round_history = []

        for r in range(1, rounds + 1):
            logger.info(f"=== Robust Training Round {r}/{rounds} ===")

            # Step 1: Train detector on current graph
            self.model.train()
            for epoch in range(1, epochs_per_round + 1):
                base_trainer.train_epoch(current_graph)

            # Step 2: Freeze detector
            for p in self.model.parameters():
                p.requires_grad = False
            self.model.eval()

            # Clean evaluation
            clean_metrics = base_trainer.evaluate(current_graph, mask=current_graph.test_mask)

            # Step 3: Generate adversarial perturbations on training fraud nodes
            logger.info("Generating adversarial structuring perturbations...")
            if self.attack_method == "gradient":
                attacker = ConstrainedGradientAttacker(detector=self.model)
            else:
                attacker = HeuristicAttacker(detector=self.model, seed=self.seed + r)

            train_fraud_nodes = (current_graph.train_mask & (current_graph.y == 1)).nonzero(as_tuple=True)[0].cpu().numpy()
            sample_size = max(1, int(len(train_fraud_nodes) * adversarial_ratio))
            targets = train_fraud_nodes[:sample_size]

            hardened_graph = current_graph.clone()
            evaded_count = 0

            for node in targets:
                hardened_graph, summary = attacker.attack_node(
                    hardened_graph,
                    target_node=int(node),
                    max_steps=6
                )
                if summary["evaded"]:
                    evaded_count += 1

            # Step 4: Evaluate attack on test set to measure current adversarial recall
            test_fraud_nodes = (current_graph.test_mask & (current_graph.y == 1)).nonzero(as_tuple=True)[0].cpu().numpy()
            test_attack_graph = current_graph.clone()
            for node in test_fraud_nodes:
                test_attack_graph, _ = attacker.attack_node(
                    test_attack_graph,
                    target_node=int(node),
                    max_steps=6
                )

            adv_metrics = base_trainer.evaluate(test_attack_graph, mask=test_attack_graph.test_mask)
            robust_metrics = compute_robustness_metrics(clean_metrics, adv_metrics)

            logger.info(
                f"Round {r} Evaluation -> Clean Recall: {robust_metrics['clean_recall']:.4f}, "
                f"Adversarial Recall: {robust_metrics['adversarial_recall']:.4f}, "
                f"Recall Drop: {robust_metrics['recall_drop']:.4f}, "
                f"Hard-Negative FPR: {robust_metrics.get('hard_negative_fpr', 0.0):.4f}"
            )

            round_record = {
                "round": r,
                "clean_recall": robust_metrics["clean_recall"],
                "adversarial_recall": robust_metrics["adversarial_recall"],
                "recall_drop": robust_metrics["recall_drop"],
                "f1": clean_metrics["f1"],
                "hard_negative_fpr": robust_metrics.get("hard_negative_fpr", 0.0)
            }
            round_history.append(round_record)

            # Step 5: Unfreeze detector and update graph with hardened examples
            for p in self.model.parameters():
                p.requires_grad = True

            current_graph = hardened_graph

        # Final save
        save_checkpoint(
            self.model.state_dict(),
            self.output_dir / "best_model.pt",
            metadata={"round_history": round_history}
        )
        save_json({"rounds": round_history}, self.output_dir / "robust_training_history.json")
        logger.info("Adversarial robust training completed successfully!")

        return {
            "round_history": round_history,
            "final_robust_metrics": round_history[-1] if round_history else {}
        }
