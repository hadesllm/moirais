"""Tests for moirais.fn.reldi — Reliability diagram values."""
import numpy as np
import pytest
from moirais.fn.reldi import reldi


@pytest.fixture()
def data():
    rng = np.random.default_rng(25)
    n = 500
    p = rng.uniform(0.1, 0.9, n)
    y = rng.binomial(1, p).astype(float)
    return y, p


def test_keys(data):
    r = reldi(*data)
    for k in ("bin_centers", "fraction_positive", "mean_confidence",
              "bin_counts", "calibration_gaps", "overconfident_bins",
              "underconfident_bins", "n", "method"):
        assert k in r


def test_bin_count(data):
    r = reldi(*data, n_bins=10)
    assert len(r["bin_centers"]) == 10


def test_fraction_positive_in_01(data):
    r = reldi(*data)
    valid = ~np.isnan(r["fraction_positive"])
    assert np.all(r["fraction_positive"][valid] >= 0)
    assert np.all(r["fraction_positive"][valid] <= 1)


def test_calibration_gaps_nonneg(data):
    r = reldi(*data)
    valid = ~np.isnan(r["calibration_gaps"])
    assert np.all(r["calibration_gaps"][valid] >= 0)


def test_perfect_calibration_low_gap():
    """Perfectly calibrated probs should have near-zero gaps."""
    rng = np.random.default_rng(99)
    p = np.linspace(0.05, 0.95, 500)
    y = rng.binomial(1, p).astype(float)
    r = reldi(y, p, n_bins=10)
    valid = ~np.isnan(r["calibration_gaps"])
    assert np.nanmean(r["calibration_gaps"][valid]) < 0.2


def test_quantile_strategy(data):
    r = reldi(*data, strategy="quantile")
    assert r["method"].endswith("quantile")


def test_n_correct(data):
    y, p = data
    assert reldi(*data)["n"] == len(y)


def test_cheatsheet():
    from moirais.fn.reldi import cheatsheet
    assert len(cheatsheet()) > 0
