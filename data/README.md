# AdaMule Data Directory

This directory contains synthetic transaction and graph datasets calibrated for money-mule detection research.

## Structure
- `raw/`: Raw generated CSVs of transactions and entity tables (Accounts, Merchants, Devices, IPs).
- `processed/`: Preprocessed graph objects, feature tensors, and train/val/test temporal splits.
- `synthetic/`: Exported synthetic graph benchmarks for reproducible experiment runs.

## Research & Ethical Boundary Note
Per Specification Section 6, 43, and 44:
All data generated within this repository is strictly **synthetic** and calibrated against publicly documented mule-ring typologies (FATF, RBI, FinCEN advisories) and legitimate small-merchant aggregator patterns.
It does **not** contain or represent actual bank or customer data.
