# 02. How It Works (Architecture & Dataflow)

This document explains the complete engineering pipeline of AdaMule, from data generation to deep learning training.

---

## 1. High-Level Closed-Loop Architecture

```
+-------------------------------------------------------------------------+
|                  1. SYNTHETIC TRANSACTION ENGINE                        |
|   Generates realistic heterogeneous entities (Retail, Students,         |
|   Merchants, Devices, IPs) across Scenarios A through I.                |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                   2. GRAPH BUILDER & TEMPORAL SPLIT                     |
|   Constructs heterogeneous directed graph with temporal timestamps.     |
|   Strict chronological train -> val -> test partition (No leakage!).   |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                   3. GRAPH NEURAL NETWORK DETECTORS                     |
|   - Baselines: GCN, GAT, CARE-GNN, FRAUDRE, Temporal GNN                |
|   - Primary Model: AdaMule (Fourier Time Encoder + Legitimacy Module)   |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|            4. FINANCIALLY CONSTRAINED ADVERSARIAL ATTACKER              |
|   Simulates smart criminals: Smurfing, Merchant Camouflage,             |
|   Layering Hops, and Temporal Jitter (Heuristic, Gradient, PPO RL).     |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|              5. MIN-MAX ADVERSARIAL ROBUST RETRAINING                   |
|   min_theta max_delta L(theta; G + delta)                               |
|   The detector learns to anticipate and neutralize evasive attacks.     |
+-------------------------------------------------------------------------+
                                    |
                                    v
+-------------------------------------------------------------------------+
|                  6. EMPIRICAL EVALUATION & WAR ROOM                     |
|   - Benchmark comparison CSVs & publication charts                      |
|   - Interactive Web Cyber War Room & 3D WebGL Globe                     |
+-------------------------------------------------------------------------+
```

---

## 2. Step-by-Step Breakdown of Each Phase

### Step 1: Calibrated Synthetic Data Generation (`src/adamule/data/`)
Because real banking transaction data contains strict personal privacy (PII) restrictions, AdaMule includes a mathematically calibrated synthetic data generator that mirrors real-world transaction distributions:
- **Entities**: Retail accounts, college student accounts, registered merchant accounts, shared phone devices, and IP addresses.
- **Scenarios A through I**:
  * *Scenario A*: Normal everyday peer-to-peer (P2P) transfers.
  * *Scenario B*: **Legitimate Merchant Aggregator** (The critical hard negative: *Sharma Kirana*).
  * *Scenario C*: Irregular / seasonal wholesale merchant.
  * *Scenario D*: Simple 3-hop linear mule ring ($A 	o B 	o C 	o 	ext{Cashout}$).
  * *Scenario E*: Fan-in aggregation mule.
  * *Scenario F*: Fan-out dispersal mule.
  * *Scenario G*: Multi-hop layered network.
  * *Scenario H*: **Camouflaged mule network** (criminals injecting micro-payments to merchants to fool the AI).
  * *Scenario I*: Adaptive evasive structuring.

### Step 2: Leak-Free Temporal Graph Partitioning (`src/adamule/data/dataset.py`)
In real life, an AI cannot look into the future. Many bad academic papers cheat by randomly shuffling transactions across train and test sets, causing "future data leakage".
AdaMule uses **strictly chronological temporal splitting**:
$$	ext{Train Period} \le 	ext{Validation Period} \le 	ext{Test Period}$$
This ensures 100% honest evaluation.

### Step 3: Deep Graph Neural Network Models (`src/adamule/models/`)
AdaMule implements and benchmarks 6 distinct AI models:
1. **GCN (Graph Convolutional Network)**: Classic message-passing baseline.
2. **GAT (Graph Attention Network)**: Uses attention weights to focus on important neighbor nodes.
3. **CARE-GNN**: Relation-aware fraud detector with neighbor reinforcement.
4. **FRAUDRE**: Fraud detection model with camouflage-resistant embeddings.
5. **Temporal GNN**: Uses time intervals to filter bursts.
6. **AdaMule (Primary Model)**:
   - **Continuous Fourier Time Encoder**: Computes dynamic edge representations:
     $$\phi(\Delta t) = [\cos(\omega_1 \Delta t), \sin(\omega_1 \Delta t), \dots]$$
     This allows AdaMule to detect rapid 5-minute pass-through transfers even if fraudsters add delay jitter!
   - **Legitimacy Regularizer Head**: An auxiliary loss that penalizes the model if it confuses an irregular merchant with a mule ring.

### Step 4: The Adversarial Attack Engine (`src/adamule/attacks/`)
To test if the AI can be fooled, AdaMule implements an adversarial attack simulator under real financial constraints:
- **Financial Constraints Validator**:
  * Total balance must be conserved (criminals cannot create money out of thin air).
  * Transfers cannot flow backwards in time.
  * Individual transfer amounts must stay within realistic boundaries.
- **Attack Stages**:
  * *Stage 1*: Random valid perturbations.
  * *Stage 2 (Heuristic)*: Smurfing amounts below ₹50,000 threshold and injecting decoy edges into merchants.
  * *Stage 3 (Gradient)*: Uses neural loss gradients to identify the exact weakest edges in the graph.
  * *Stage 4 (Reinforcement Learning)*: A Gymnasium environment (`StructuringAttackEnv`) where a PPO agent learns optimal laundering policies.

### Step 5: Min-Max Adversarial Retraining (`src/adamule/training/`)
To make AdaMule immune to these attacks, it uses game-theoretic min-max training:
$$\min_{	heta} \max_{\delta \in \Delta} \mathcal{L}(	heta; G + \delta)$$
1. The attacker creates the most evasive perturbations ($\max$).
2. The detector trains its weights $	heta$ to correctly identify the evasive subgraphs ($\min$).
3. Result: The hardened AdaMule model maintains high detection rates even when criminals actively try to trick it!
