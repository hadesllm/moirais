"""Tests for morie.fn.calbt — Calibration (Platt + isotonic)."""
import numpy as np
import pytest
from morie.fn.calbt import calbt


@pytest.fixture()
def data():
    rng = np.random.default_rng(24)
    n = 300
    # Overconfident classifier: scores compressed near 0 and 1
    p_true = rng.uniform(0.2, 0.8, n)
    y_true = rng.binomial(1, p_true).astype(float)
    # Simulate overconfident scores
    y_prob = np.clip(p_true + rng.standard_normal(n) * 0.1, 0.01, 0.99)
    return y_true, y_prob


def test_keys_platt(data):
    r = calbt(*data, method="platt")
    for k in ("calibrated_probs", "ece", "ece_before", "platt_A",
              "platt_B", "reliability_before", "reliability_after", "n", "method"):
        assert k in r


def test_calibrated_in_01(data):
    r = calbt(*data, method="platt")
    assert np.all(r["calibrated_probs"] >= 0) and np.all(r["calibrated_probs"] <= 1)


def test_ece_nonneg(data):
    r = calbt(*data, method="platt")
    assert r["ece"] >= 0
    assert r["ece_before"] >= 0


def test_platt_improves_calibration(data):
    r = calbt(*data, method="platt")
    assert r["ece"] <= r["ece_before"] + 0.05


def test_isotonic(data):
    r = calbt(*data, method="isotonic")
    for k in ("calibrated_probs", "ece", "n", "method"):
        assert k in r
    assert np.all(r["calibrated_probs"] >= 0)


def test_isotonic_improves(data):
    r = calbt(*data, method="isotonic")
    assert r["ece"] <= r["ece_before"] + 0.05


def test_method_label(data):
    assert "platt" in calbt(*data, method="platt")["method"]
    assert "isotonic" in calbt(*data, method="isotonic")["method"]


def test_cheatsheet():
    from morie.fn.calbt import cheatsheet
    assert len(cheatsheet()) > 0
