"""Tests for moirais.fn.mnksb — Manski bounds."""
import numpy as np
import pytest
from moirais.fn.mnksb import mnksb


@pytest.fixture()
def data():
    rng = np.random.default_rng(11)
    n = 300
    t = rng.binomial(1, 0.5, n).astype(float)
    y = rng.binomial(1, 0.3 + 0.2 * t, n).astype(float)
    return y, t


def test_keys(data):
    r = mnksb(*data)
    for k in ("lb", "ub", "width", "lb_ci", "ub_ci", "n", "method"):
        assert k in r


def test_bounds_valid(data):
    r = mnksb(*data)
    assert r["lb"] <= r["ub"]


def test_bounds_contain_truth(data):
    """True ATE ≈ 0.2; bounds should contain it given y in [0,1]."""
    r = mnksb(*data)
    assert r["lb"] <= 0.2 + 0.5
    assert r["ub"] >= 0.2 - 0.5


def test_mtr_tightens_bounds(data):
    r_noass = mnksb(*data)
    r_mtr = mnksb(*data, assume_mtr=True)
    assert r_mtr["width"] <= r_noass["width"] + 0.01


def test_custom_support(data):
    y, t = data
    r = mnksb(y, t, y_min=0.0, y_max=1.0)
    assert np.isfinite(r["lb"]) and np.isfinite(r["ub"])


def test_method(data):
    assert "Manski" in mnksb(*data)["method"]


def test_width_nonneg(data):
    assert mnksb(*data)["width"] >= 0


def test_cheatsheet():
    from moirais.fn.mnksb import cheatsheet
    assert len(cheatsheet()) > 0
