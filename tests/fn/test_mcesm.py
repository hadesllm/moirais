"""Tests for morie.fn.mcesm — Maximum calibration error."""
import numpy as np
import pytest
from morie.fn.mcesm import mcesm


@pytest.fixture()
def data():
    rng = np.random.default_rng(27)
    n = 500
    p = rng.uniform(0.1, 0.9, n)
    y = rng.binomial(1, p).astype(float)
    return y, p


def test_keys(data):
    r = mcesm(*data)
    for k in ("mce", "worst_bin_idx", "worst_bin_center", "worst_acc",
              "worst_conf", "bin_gaps", "bin_counts", "n", "method"):
        assert k in r


def test_mce_nonneg(data):
    assert mcesm(*data)["mce"] >= 0 or np.isnan(mcesm(*data)["mce"])


def test_mce_geq_ece():
    """MCE should be >= ECE for the same data."""
    rng = np.random.default_rng(55)
    p = rng.uniform(0.1, 0.9, 500)
    y = rng.binomial(1, p).astype(float)
    r_mce = mcesm(y, p)
    from morie.fn.ecesm import ecesm
    r_ece = ecesm(y, p)
    assert r_mce["mce"] >= r_ece["ece"] - 1e-8


def test_worst_bin_valid(data):
    r = mcesm(*data)
    if r["worst_bin_idx"] >= 0:
        assert 0.0 <= r["worst_bin_center"] <= 1.0


def test_perfect_calibration_low_mce():
    rng = np.random.default_rng(66)
    p = np.linspace(0.05, 0.95, 1000)
    y = rng.binomial(1, p).astype(float)
    r = mcesm(y, p, n_bins=10)
    assert np.isnan(r["mce"]) or r["mce"] < 0.3


def test_quantile_strategy(data):
    r = mcesm(*data, strategy="quantile")
    assert "quantile" in r["method"]


def test_n_correct(data):
    y, p = data
    assert mcesm(*data)["n"] == len(y)


def test_cheatsheet():
    from morie.fn.mcesm import cheatsheet
    assert len(cheatsheet()) > 0
