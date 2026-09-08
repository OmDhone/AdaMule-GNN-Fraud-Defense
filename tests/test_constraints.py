"""Test suite verifying financial constraints enforcement and perturbation budgets."""

import pytest
from adamule.attacks.constraints import FinancialConstraintsEngine


def test_amount_constraints():
    engine = FinancialConstraintsEngine()
    
    # Valid amount
    valid, msg = engine.validate_amount(500.0)
    assert valid
    
    # Below minimum
    valid_low, msg_low = engine.validate_amount(5.0)
    assert not valid_low
    assert "below minimum" in msg_low.lower()
    
    # Above maximum
    valid_high, msg_high = engine.validate_amount(250000.0)
    assert not valid_high
    assert "exceeds maximum" in msg_high.lower()


def test_fund_conservation_split():
    engine = FinancialConstraintsEngine()
    
    # Valid split: 1000 into 300, 300, 400
    valid, msg = engine.validate_structuring_split(1000.0, [300.0, 300.0, 400.0])
    assert valid
    
    # Invalid split: fund creation / loss beyond tolerance
    invalid, msg_inv = engine.validate_structuring_split(1000.0, [200.0, 200.0])
    assert not invalid
    assert "fund conservation violated" in msg_inv.lower()
    
    # Invalid split element (below min threshold)
    invalid_elem, msg_elem = engine.validate_structuring_split(1000.0, [995.0, 5.0])
    assert not invalid_elem


def test_budget_constraints():
    engine = FinancialConstraintsEngine()
    
    # Within budget
    valid, _ = engine.validate_budget(current_step=3, edges_added=2, edges_modified=4)
    assert valid
    
    # Exceeded added edges budget
    inv_add, msg_add = engine.validate_budget(current_step=5, edges_added=15, edges_modified=2)
    assert not inv_add
    assert "added edges" in msg_add.lower()
    
    # Exceeded max steps
    inv_step, msg_step = engine.validate_budget(current_step=25, edges_added=2, edges_modified=2)
    assert not inv_step
    assert "steps" in msg_step.lower()


def test_temporal_causality():
    engine = FinancialConstraintsEngine()
    
    # Valid causal sequence: new tx happens after fund receipt
    valid, _ = engine.validate_temporal_causality(new_timestamp=2000.0, preceding_timestamp=1000.0)
    assert valid
    
    # Invalid: new tx attempts to forward funds before receiving them
    invalid, msg = engine.validate_temporal_causality(new_timestamp=500.0, preceding_timestamp=1000.0)
    assert not invalid
    assert "earlier" in msg.lower()
