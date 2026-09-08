# 06. Complete Codebase Directory Map

A complete reference guide explaining what every folder and file in this repository does.

```
C:\Users\dhone\OneDrive\Desktop\MLCV\
|-- configs/                        # YAML Configuration profiles
|   |-- attack.yaml                 # Budgets and action spaces for adversarial evasion
|   |-- data.yaml                   # Profiles: development, medium, research datasets
|   |-- experiments.yaml            # Master setup for Experiments A through K
|   |-- model.yaml                  # GNN architectures, learning rates, hidden dimensions
|   `-- training.yaml               # Early stopping patience, min-max retraining rounds
|
|-- data/                           # Generated synthetic graph datasets and splits
|
|-- outputs/                        # Research outputs, charts, and reports
|   |-- figures/                    # Publication PNG figures (Recall, FPR, Ablation)
|   `-- reports/                    # benchmark_comparison.csv & final_research_report.md
|
|-- project_description/            # Comprehensive documentation suite (You are here!)
|   |-- README.md                   # Navigation index
|   |-- 01_WHAT_IS_THIS_PROJECT.md  # Real-world problem & why traditional AI fails
|   |-- 02_HOW_IT_WORKS_ARCHITECTURE.md # 13-phase architecture & mathematical formulation
|   |-- 03_ALL_FEATURES_EXPLAINED.md    # Guide to all 10 interactive features
|   |-- 04_TECHNOLOGY_STACK.md      # PyTorch, NetworkX, Three.js, etc.
|   |-- 05_STEP_BY_STEP_HOW_TO_RUN.md   # Copy-paste run commands
|   `-- 06_COMPLETE_CODEBASE_DIRECTORY_MAP.md # This file!
|
|-- scripts/                        # Executable CLI scripts
|   |-- generate_data.py            # Generates synthetic graph data
|   |-- train_baseline.py           # Trains GCN, GAT, CARE-GNN, FRAUDRE
|   |-- train_attacker.py           # Trains the Gymnasium PPO RL attacker
|   |-- run_attack.py               # Executes heuristic or gradient attacks
|   |-- train_adamule.py            # Trains AdaMule with min-max retraining
|   |-- evaluate.py                 # Computes ROC-AUC, FPR, recall drops
|   |-- run_experiment.py           # Master end-to-end benchmark runner
|   `-- launch_war_room.py          # Auto-opening local HTTP server for the web war room
|
|-- src/adamule/                    # Core Python Package Source Code
|   |-- attacks/                    # Adversarial Evasion Simulator
|   |   |-- actions.py              # Action primitives (Smurfing, Camouflage, Hops)
|   |   |-- constraints.py          # Financial rules engine (Conservation, Causality)
|   |   |-- environment.py          # Gymnasium MDP environment (StructuringAttackEnv)
|   |   |-- gradient_attack.py      # White-box gradient perturbation attack
|   |   |-- heuristic.py            # Rule-based structuring attacker
|   |   |-- reward.py               # Attacker evasion reward function
|   |   `-- rl_attacker.py          # PPO policy gradient training wrapper
|   |
|   |-- data/                       # Data Pipeline
|   |   |-- dataset.py              # Chronological leak-free temporal splitting
|   |   |-- features.py             # Structural, temporal, & legitimacy feature extraction
|   |   |-- generator.py            # Accounts, Merchants, Devices, IPs entity generator
|   |   |-- graph_builder.py        # Constructs PyTorch geometric graph tensors
|   |   `-- scenarios.py            # Synthetic Scenarios A through I generator
|   |
|   |-- evaluation/                 # Metrics & Compliance Forensics
|   |   |-- calibration.py          # Probability calibration & reliability curves
|   |   |-- comparisons.py          # Multi-model benchmarking comparison tables
|   |   |-- forensics.py            # Automated SAR legal narrative generator
|   |   |-- metrics.py              # Precision, Recall, F1, PR-AUC, FPR calculation
|   |   |-- reports.py              # Markdown & CSV report compiler
|   |   `-- robustness.py           # Adversarial robustness ratio & recall drop metrics
|   |
|   |-- models/                     # Deep Learning Graph Neural Networks
|   |   |-- adamule.py              # Primary Model: Fourier Time Encoder + Legitimacy Module
|   |   |-- care_gnn.py             # Relation-aware CARE-GNN baseline
|   |   |-- fraudre.py              # Camouflage-resistant FRAUDRE baseline
|   |   |-- gat.py                  # Multi-head Graph Attention Network
|   |   |-- gcn.py                  # Graph Convolutional Network baseline
|   |   `-- temporal.py             # Temporal Graph Network baseline
|   |
|   |-- training/                   # Model Training Loops
|   |   |-- adversarial_trainer.py  # Min-max alternating robust retraining loop
|   |   |-- early_stopping.py       # Checkpointing & patience monitor
|   |   |-- losses.py               # Combined fraud loss + legitimacy regularizer
|   |   `-- trainer.py              # Standard single-model GNN trainer
|   |
|   |-- utils/                      # Utilities
|   |   |-- config.py               # YAML configuration parser
|   |   |-- io.py                   # Atomic file reading & writing
|   |   |-- logging.py              # Formatted colored console logger
|   |   `-- seed.py                 # Deterministic global RNG seeding
|   |
|   `-- visualization/              # Streamlit & Matplotlib Plotting
|       |-- attack_plots.py         # Evasion trajectory curves
|       |-- dashboards.py           # Multi-tab Streamlit dashboard
|       |-- graph_plots.py          # Matplotlib graph visualizations
|       |-- interactive_network.py  # PyVis interactive HTML network generator
|       `-- metric_plots.py         # Ablation and FPR comparison bar charts
|
|-- tests/                          # 27 Unit Tests (100% Pass Rate)
|   |-- test_attacker.py            # Tests evasion rewards and RL environment
|   |-- test_constraints.py         # Tests amount, conservation, and causality checks
|   |-- test_data.py                # Tests entity and transaction schemas
|   |-- test_evaluation.py          # Tests metrics and hard-negative FPR formulas
|   |-- test_graph.py               # Tests graph builder and leak-free temporal split
|   |-- test_models.py              # Tests forward/backward passes for all 6 GNNs
|   `-- test_training.py            # Tests early stopping and min-max loop
|
|-- web/                            # Cyber Defense War Room Web Application
|   |-- app.js                      # 60 FPS physics graph, heist engine, & UI handlers
|   |-- audio.js                    # Web Audio API procedural sound synthesizer
|   |-- campaign.js                 # 3-level Cyber Detective Story Mode
|   |-- chaos.js                    # Chaos Engine & stress-testing shock injector
|   |-- crypto.js                   # Web3 USDT peel-chain mempool scanner
|   |-- evidence.js                 # 1-click court-admissible evidence JSON exporter
|   |-- globe.js                    # Three.js 3D WebGL Holographic Cyber Globe
|   |-- index.html                  # Main UI layout, modals, phone mockup, & marquee
|   |-- styles.css                  # Cyberpunk dark neon glassmorphic CSS styles
|   |-- three.min.js                # Offline local Three.js r128 library
|   `-- voice.js                    # Web Speech API voice-activated AI copilot
|
|-- pyproject.toml                  # Installable Python package definition
|-- requirements.txt                # Python package dependencies
|-- start_war_room.bat              # 1-click Windows batch launcher
`-- README.md                       # Root repository guide
```
