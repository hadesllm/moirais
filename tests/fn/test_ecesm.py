"""Tests for moirais.fn.ecesm — Expected calibration error."""
import numpy as np
import pytest
from moirais.fn.ecesm import ecesm


@pytest.fixture()
def data():
    rng = np.random.default_rng(26)
    n = 500
    p = rng.uniform(0.1, 0.9, n)
    y = rng.binomial(1, p).astype(float)
    return y, p


def test_keys(data):
    r = ecesm(*data)
    for k in ("ece", "bin_accs", "bin_confs", "bin_counts", "n", "n_bins", "norm", "method"):
        assert k in r


def test_ece_nonneg(data):
    assert ecesm(*data)["ece"] >= 0


def test_ece_leq1(data):
    assert ecesm(*data)["ece"] <= 1.0


def test_perfect_calibration_low_ece():
    """Near-perfect calibration should yield ECE < 0.05."""
    rng = np.random.default_rng(88)
    p = np.linspace(0.05, 0.95, 1000)
    y = rng.binomial(1, p).astype(float)
    r = ecesm(y, p, n_bins=10)
    assert r["ece"] < 0.1


def test_overconfident_high_ece():
    """All-0.9 predictions for 50/50 labels => ECE ~ 0.4."""
    rng = np.random.default_rng(77)
    y = rng.binomial(1, 0.5, 500).astype(float)
    p = np.full(500, 0.9)
    r = ecesm(y, p, n_bins=5)
    assert r["ece"] > 0.2


def test_norm2_rmsce(data):
    r1 = ecesm(*data, norm=1)
    r2 = ecesm(*data, norm=2)
    assert np.isfinite(r2["ece"])


def test_bootstrap_ci(data):
    r = ecesm(*data, ci_method="bootstrap", n_boot=100, seed=0)
    assert np.isfinite(r["ci_lower"]) and np.isfinite(r["ci_upper"])
    assert r["ci_lower"] <= r["ece"] <= r["ci_upper"] + 0.01


def test_n_bins_stored(data):
    r = ecesm(*data, n_bins=15)
    assert r["n_bins"] == 15


def test_cheatsheet():
    from moirais.fn.ecesm import cheatsheet
    assert len(cheatsheet()) > 0
