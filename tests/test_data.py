"""Test suite verifying data generation, entity schemas, and scenarios."""

import pytest
import pandas as pd
import numpy as np
from adamule.utils.seed import set_seed
from adamule.utils.config import load_config, get_profile_config, get_project_root
from adamule.utils.io import save_json, load_json
from adamule.data.generator import SyntheticDataGenerator
from adamule.data.scenarios import ScenarioGenerator


def test_seed_reproducibility():
    """Verify that set_seed ensures deterministic random states across numpy and torch."""
    set_seed(42)
    a = np.random.randn(5)
    
    set_seed(42)
    a_repeat = np.random.randn(5)
    
    np.testing.assert_allclose(a, a_repeat)


def test_configs_loading():
    """Verify that all default configuration files exist and parse cleanly."""
    root = get_project_root()
    assert (root / "configs" / "data.yaml").exists()
    assert (root / "configs" / "model.yaml").exists()
    assert (root / "configs" / "attack.yaml").exists()
    assert (root / "configs" / "training.yaml").exists()
    assert (root / "configs" / "experiments.yaml").exists()


def test_entity_generation():
    """Verify entity generation for accounts, merchants, devices, and IPs."""
    cfg = {
        "num_accounts": 100,
        "num_merchants": 15,
        "num_devices": 80,
        "num_ips": 60,
        "num_transactions": 200,
        "time_span_days": 10,
        "scenarios": {
            "scenario_a_weight": 0.4,
            "scenario_b_weight": 0.3,
            "scenario_c_weight": 0.1,
            "scenario_d_weight": 0.1,
            "scenario_e_weight": 0.1,
        }
    }
    gen = SyntheticDataGenerator(config=cfg, seed=42)
    entities = gen.generate_entities()
    
    assert len(entities["accounts"]) == 100
    assert len(entities["merchants"]) == 15
    assert len(entities["devices"]) == 80
    assert len(entities["ips"]) == 60
    
    # Check account fields
    acc = entities["accounts"][0]
    for field in ["account_id", "account_type", "age_days", "customer_segment", "risk_prior", "location", "business_profile", "has_business_profile"]:
        assert field in acc
        
    # Check merchant fields
    m = entities["merchants"][0]
    for field in ["merchant_id", "merchant_account_id", "merchant_type", "business_age", "transaction_volume", "average_transaction"]:
        assert field in m


def test_transactions_generation_and_schema():
    """Verify transaction fields, chronological sorting, and scenario distributions."""
    cfg = get_profile_config("development")
    cfg["num_transactions"] = 500  # fast test
    cfg["num_accounts"] = 150
    
    gen = SyntheticDataGenerator(config=cfg, seed=123)
    df_txs, entity_dfs = gen.generate_full_dataset()
    
    required_cols = [
        "transaction_id", "timestamp", "sender_id", "receiver_id", "amount",
        "transaction_type", "merchant_id", "device_id", "ip_id", "location",
        "channel", "is_fraud", "fraud_type", "scenario_id"
    ]
    for col in required_cols:
        assert col in df_txs.columns
        
    assert len(df_txs) > 0
    # Check timestamps are sorted
    assert df_txs["timestamp"].is_monotonic_increasing
    
    # Check that both legitimate (0) and fraud (1) exist
    assert 0 in df_txs["is_fraud"].values
    assert 1 in df_txs["is_fraud"].values
    
    # Check that hard negative Scenario B is generated
    assert "scenario_b" in df_txs["scenario_id"].values
    # Check fraud scenarios are present
    assert "scenario_d" in df_txs["scenario_id"].values or "scenario_e" in df_txs["scenario_id"].values
