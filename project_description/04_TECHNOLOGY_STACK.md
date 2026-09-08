# 04. Technology Stack & Libraries Used

AdaMule is engineered with a modern, CPU-friendly, zero-bloat technology stack.

---

## 1. Machine Learning & Backend (Python 3.12)

| Library / Tool | Version | Purpose in AdaMule |
| :--- | :--- | :--- |
| **PyTorch (CPU)** | `2.14.0+cpu` | Deep learning backend. Implements message-passing layers, Fourier continuous-time encodings, and gradient-based attacks without requiring CUDA or C++ compilers. |
| **NetworkX** | `3.4.2` | Graph algorithms, topological traversals, ego-network extraction, and centrality computation. |
| **NumPy** | `1.26.4` | High-performance matrix mathematics, tensor manipulations, and statistical sampling. |
| **Pandas** | `2.2.3` | Transaction tabular data processing, temporal sorting, and CSV export. |
| **Scikit-Learn** | `1.5.2` | Evaluation metrics: ROC-AUC, PR-AUC, Confusion Matrices, Precision, Recall, and F1-Scores. |
| **Gymnasium** | `1.0.0` | Reinforcement Learning MDP environment (`StructuringAttackEnv`) for Stage 4 RL attacker. |
| **Stable-Baselines3**| `2.4.0` | PPO (Proximal Policy Optimization) agent training against the graph detector. |
| **PyYAML** | `6.0.2` | Hierarchical configuration management (`configs/data.yaml`, `configs/model.yaml`, etc.). |
| **Pytest** | `9.1.1` | Automated test suite verifying 27 distinct unit and integration test cases. |
| **Streamlit** | `1.42.0` | Multi-tab research and data exploration dashboard. |
| **Matplotlib / Seaborn**| `3.10.0` | Publication-quality research figures, ablation bar charts, and calibration plots. |

---

## 2. Interactive Web Frontend (Zero-Dependency Vanilla Web Stack)

| Technology | Purpose in AdaMule |
| :--- | :--- |
| **Three.js (r128)** | 3D WebGL graphics engine powering the interactive rotating Cyber Globe, atmospheric halos, starfields, and 3D ballistic arcs. |
| **HTML5 Canvas 2D** | 60 FPS force-directed physics graph with dynamic repulsion, spring tension, and animated cash particle swarms. |
| **Web Audio API** | Procedural sound synthesizer. Creates real-time cash register chimes, laser pulses, alarm sirens, and freeze shimmers with **zero external audio files**! |
| **Web Speech API** | Native browser speech recognition and synthetic speech synthesis for hands-free voice control. |
| **CSS3 Glassmorphism**| High-tech dark cyberpunk UI with blur filters, glowing neon borders, responsive grids, and print-ready styles. |
| **Python `http.server`**| Lightweight local web server powering the command center on port 8080 with auto-browser launch. |
