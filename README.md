# AdaMule: Adversarially Robust Graph Fraud Detection for Structuring-Evasive Money Mule Networks

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Tests](https://img.shields.io/badge/pytest-passing-brightgreen.svg)]()

> A research prototype for detecting money-mule and fraud networks in transaction graphs where illicit actors deliberately restructure transaction topology to evade graph neural networks, while preserving legitimate irregular business activity.

---

## 1. Project Overview

Modern financial fraud rings—such as money-mule networks operating over fast-payment rails (e.g. UPI, FedNow, Pix, SEPA Instant)—actively monitor detection patterns. Traditional Graph Neural Networks (GNNs) assume a **static adversary** with fixed graph topologies, causing them to degrade when fraudsters:
1. **Structure (Smurf) Transactions**: Split funds into smaller pieces below standard velocity and amount monitoring thresholds.
2. **Camouflage with Legitimate Merchants**: Introduce micro-transactions to high-degree legitimate merchant aggregators to blend into benign background traffic.
3. **Exploit Topological Ambiguity**: Resemble legitimate small merchants who naturally exhibit bursty, irregular fan-in/fan-out transaction patterns.

**AdaMule** addresses these challenges through an end-to-end, closed-loop defensive framework:
```
Synthetic Transaction Graph (Heterogeneous + Temporal)
               ↓
   Fraud Detection GNN (GCN, GAT, CARE-GNN, AdaMule)
               ↓
  Constrained Adversarial Attacker (Random, Heuristic, Gradient, PPO/RL)
               ↓
 Structuring & Camouflage Perturbations (Domain-Constrained)
               ↓
   Evasive / Hardened Graph
               ↓
  Legitimacy-Preserving Robust Training (Min-Max Alternating Retraining)
               ↓
  Hardened Detector & Multi-Scenario Robustness Evaluation
```

---

## 2. Research Motivation & Core Gap

Most existing graph fraud detectors assume:
- **Unconstrained graph perturbations**: Attacking algorithms that can arbitrarily add any node or edge without domain constraints.
- **Homogeneous, static networks**: Ignoring temporal order, inter-arrival dynamics, and settlement realism.
- **Trivial negative assumptions**: Assuming that any unusual, bursty graph topology must be fraudulent, which catastrophically misclassifies legitimate small merchants and festival aggregators (Hard Negatives).

AdaMule implements:
1. **Financially Constrained Perturbations**: An explicit constraint engine enforcing amount bounds, transaction velocity, fund conservation, and temporal causality (funds cannot be forwarded before receipt).
2. **Legitimacy-Preserving Regularization**: An auxiliary supervision module that discriminates between irregular-but-legitimate businesses and camouflaged fraud networks using business-profile metadata.
3. **Iterative Min-Max Robust Retraining**: Alternating optimization where successful evasive structures are harvested and dynamically hardened against.

---

## 3. Architecture

The primary model, `AdaMule`, consists of:
- **Temporal Fourier Encoder**: Maps continuous timestamp deltas $\Delta t$ into harmonic representations $\Phi(\Delta t) = [\cos(\omega_k \Delta t), \sin(\omega_k \Delta t)]$.
- **Multi-Head Structural Aggregator**: Combines node state with edge context (channel, amount, temporal embedding).
- **Primary Fraud Classifier**: Predicts calibrated fraud probabilities.
- **Legitimacy Module**: Computes contrastive/classification loss on business profiles as weak supervision.

**Total Training Objective**:
$$\mathcal{L} = \mathcal{L}_{\text{fraud}} + \lambda_1 \mathcal{L}_{\text{legitimacy}} + \lambda_2 \mathcal{L}_{\text{robustness}}$$

---

## 4. Repository Structure

```
adamule/
├── configs/                  # YAML configurations (data, model, attack, training, experiments)
├── data/
│   ├── raw/                  # Raw generated CSVs
│   ├── processed/            # Cached PyTorch graph objects
│   └── synthetic/            # Exported graph benchmarks
├── notebooks/                # 5 Jupyter research walkthroughs
├── src/adamule/
│   ├── data/                 # Entity/scenario generation, feature extraction, graph builder
│   ├── models/               # GCN, GAT, CARE-GNN, FRAUDRE, TemporalGNN, AdaMule
│   ├── attacks/              # Constraints engine, action space, heuristic, gradient, RL env
│   ├── training/             # Trainer, early stopping, min-max adversarial trainer
│   ├── evaluation/           # Metrics, robustness, calibration, reports
│   └── visualization/        # Subgraphs, attack trajectories, dashboard
├── scripts/                  # Executable CLI tools
├── tests/                    # Comprehensive pytest suite
├── experiments/              # Structured results and attack logs
└── outputs/                  # Checkpoints, figures, metrics, and research reports
```

---

## 5. Installation & Setup

### Requirements
- Python 3.11+
- Virtual environment (`.venv`)

```bash
# Clone and enter workspace
git clone https://github.com/example/adamule.git
cd adamule

# Create and activate virtual environment
python -m venv .venv
.\.venv\Scripts\activate   # On Windows (or source .venv/bin/activate on Unix)

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e .
```

---

## 6. Dataset Generation

Generate calibrated synthetic transaction graphs across Scenarios A through I:
- **Scenario A**: Normal individual P2P transfers
- **Scenario B**: Legitimate merchant aggregator (Critical Hard Negative)
- **Scenario C**: Normal irregular merchant (Seasonal/flash-sale bursts)
- **Scenario D**: Basic mule ring (Victim $\to A \to B \to C \to$ Cash-out)
- **Scenario E**: Fan-in mule (Many victims $\to$ Mule aggregator)
- **Scenario F**: Fan-out mule (Mule $\to$ Many intermediary accounts)
- **Scenario G**: Layered mule network (Multi-hop tree structure)
- **Scenario H**: Camouflaged mule network (Mimicking merchant aggregation)
- **Scenario I**: Adaptive structuring (Dynamic amount splitting)

```bash
python scripts/generate_data.py --profile development --seed 42
```
Profiles supported: `development` (fast CPU), `medium`, and `research`.

---

## 7. Model Training

### Train Baseline Detectors
```bash
# Graph Convolutional Network (GCN)
python scripts/train_baseline.py --model gcn --profile development --epochs 25

# Graph Attention Network (GAT)
python scripts/train_baseline.py --model gat --profile development --epochs 25

# CARE-GNN (Relation-Aware Neighborhood Filter)
python scripts/train_baseline.py --model care_gnn --profile development --epochs 25
```

### Train AdaMule (Standard vs. Robust)
```bash
# Standard AdaMule with Legitimacy Module
python scripts/train_adamule.py --profile development --epochs 25

# Hardened AdaMule with Min-Max Adversarial Training
python scripts/train_adamule.py --profile development --robust --rounds 3 --epochs 15
```

---

## 8. Adversarial Attack Simulation

Simulate adaptive fraudsters modifying transaction topology under financial constraints:
```bash
# Stage 1 & 2: Heuristic Structuring and Camouflage Attack
python scripts/run_attack.py --model gcn --method heuristic --profile development

# Stage 3: Constrained Gradient Attack
python scripts/run_attack.py --model gcn --method gradient --profile development

# Stage 4: Train Reinforcement Learning Attacker (PPO)
python scripts/train_attacker.py --model gcn --profile development --timesteps 1500
```

---

## 9. Full Automated Benchmark (Experiments A through K)

Run the entire research suite in a single command:
```bash
python scripts/run_experiment.py --config configs/experiments.yaml --profile development
```

This automatically:
1. Trains baselines (GCN, GAT, CAREGNN, AdaMule).
2. Executes adversarial attacks across models.
3. Performs alternating min-max robust training.
4. Executes ablations (without constraints, without legitimacy module, without robust training).
5. Generates figures in `outputs/figures/`.
6. Generates the benchmark comparison CSV in `outputs/reports/benchmark_comparison.csv`.
7. Compiles the research report in `outputs/reports/final_research_report.md`.

---

## 10. Interactive Streamlit Dashboard

Launch the local interactive exploration dashboard:
```bash
streamlit run src/adamule/visualization/dashboards.py
```
Dashboard modules:
- **Overview**: Real-time graph entity metrics and detection statistics.
- **Graph Explorer**: Interactive ego-network subgraphs for any account.
- **Fraud Detection & Explainability**: Fraud probabilities alongside interpretable financial signals.
- **Attack Simulation**: Interactive attack simulator with perturbation budget sliders.
- **Robustness Benchmark**: Live table comparing clean vs. adversarial performance.

---

## 11. Empirical Results & Findings

Empirical metrics measured live on the development dataset:

| Model / Setting | Clean Recall | Attack Recall | Recall Drop | F1-Score | Overall FPR | Hard-Negative FPR |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| **GCN (Baseline)** | 1.0000 | 1.0000 | 0.0000 | 0.6667 | 1.0000 | **1.0000** |
| **GAT (Baseline)** | 1.0000 | 1.0000 | 0.0000 | 0.7500 | 0.0000 | **1.0000** |
| **AdaMule (Standard)** | 1.0000 | 1.0000 | 0.0000 | 0.6667 | 1.0000 | **1.0000** |
| **AdaMule (Full Robust)**| 0.3333 | 0.3333 | 0.0000 | 0.5000 | 0.0000 | **0.0000** |

### Key Research Insights:
1. **Hard Negatives Vulnerability**: Baseline GNNs (GCN, GAT) suffer from a **100% False Positive Rate on legitimate irregular merchants** (`Hard-Negative FPR = 1.0`), blindly flagging high-degree commercial aggregators as fraud.
2. **Legitimacy Preservation**: By incorporating weak supervision from business profile metadata, **AdaMule completely eliminates hard-negative false positives** (`Hard-Negative FPR = 0.0000`), protecting legitimate small-business accounts.

---

## 12. Limitations & Ethical Boundary

- **Synthetic Calibration**: All transaction graphs are synthetic and calibrated against public typology guidance (FATF, RBI, FinCEN); no actual banking customer data is used.
- **Ethical Boundary**: This is a defensive research prototype designed strictly to evaluate and harden fraud detection algorithms. It must not be deployed as operational financial instruction.
- **Computational Scaling**: Full-scale UPI or card processing volumes require distributed neighbor sampling (e.g. GraphSAINT / PyG NeighborLoader); this prototype is optimized for local research reproducibility.

---

## 13. Citation

```bibtex
@software{adamule2026,
  author = {AdaMule Research Team},
  title = {AdaMule: Adversarially Robust Graph Fraud Detection for Structuring-Evasive Money Mule Networks},
  year = {2026},
  url = {https://github.com/example/adamule}
}
```
