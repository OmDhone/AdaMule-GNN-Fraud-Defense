# 05. Step-by-Step Run Guide

Follow these simple, copy-paste instructions to run any part of the project on your Windows machine.

---

## 🎮 Option 1: Launch the Interactive Cyber War Room (Recommended!)

This is the most impressive, fun, and visual way to experience AdaMule.

### Method A: The 1-Click Batch File
1. Open your project folder: `C:\Users\dhone\OneDrive\Desktop\MLCV`
2. **Double-click** the file:
   ```text
   start_war_room.bat
   ```
3. Your browser will automatically open to **`http://localhost:8080`**!

---

### Method B: From PowerShell Terminal
1. Open PowerShell in `C:\Users\dhone\OneDrive\Desktop\MLCV`.
2. Run this command:
   ```powershell
   .\.venv\Scripts\python scripts/launch_war_room.py --port 8080
   ```
3. Open your web browser and go to:
   👉 **`http://localhost:8080`**

---

## 🧠 Option 2: Run the Full Machine Learning Benchmark Pipeline

If you want to train the models from scratch, run adversarial attacks, and generate publication-ready research reports:

Run in PowerShell:
```powershell
.\.venv\Scripts\python scripts/run_experiment.py --config configs/experiments.yaml --profile development
```

### What This Does:
1. Generates synthetic transaction scenarios A through I.
2. Trains Baseline GCN, GAT, and CARE-GNN models.
3. Runs heuristic and gradient evasion attacks.
4. Trains AdaMule with min-max adversarial retraining.
5. Exports:
   * Benchmark Comparison Table: `outputs/reports/benchmark_comparison.csv`
   * Final Research Report: `outputs/reports/final_research_report.md`
   * Visual Figures: `outputs/figures/clean_vs_adversarial_recall.png`, `outputs/figures/hard_negative_fpr_comparison.png`, `outputs/figures/ablation_study.png`.

---

## 🧪 Option 3: Run the Automated Unit Test Suite

To verify that all 27 unit tests pass with 100% success rate:

Run in PowerShell:
```powershell
.\.venv\Scripts\pytest -v
```
*(Executes in ~7 seconds and verifies data integrity, constraints, models, and training loops).*

---

## 📊 Option 4: Launch the Streamlit Multi-Tab Research Dashboard

To launch the multi-tab forensic exploration dashboard in Streamlit:

Run in PowerShell:
```powershell
.\.venv\Scripts\streamlit run src/adamule/visualization/dashboards.py --server.headless true
```
Then open:
👉 **`http://localhost:8501`**

---

## 🛠️ Troubleshooting & FAQ

* **Q: The browser says "Unable to connect"?**
  * Make sure `python scripts/launch_war_room.py` is running in your terminal.
* **Q: Microphone doesn't work for Voice AI?**
  * When you click `🎙️ Voice AI`, make sure to click **"Allow"** on your browser's microphone permission popup.
* **Q: How do I stop the local server?**
  * In the terminal running the server, press `Ctrl + C`.
