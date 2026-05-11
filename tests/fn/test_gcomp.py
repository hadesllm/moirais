"""Tests for morie.fn.gcomp — G-computation time-varying."""
import numpy as np
import pytest
from morie.fn.gcomp import gcomp


@pytest.fixture()
def data():
    rng = np.random.default_rng(6)
    n = 100
    K = 2
    T_seq = rng.binomial(1, 0.5, (n, K)).astype(float)
    L_seq = rng.standard_normal((n, K, 1))
    Y = 1.5 * T_seq[:, -1] + 0.3 * L_seq[:, -1, 0] + rng.standard_normal(n) * 0.5
    return Y, T_seq, L_seq


def test_keys(data):
    r = gcomp(*data, n_boot=50)
    for k in ("ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert k in r


def test_ate_finite(data):
    r = gcomp(*data, n_boot=50)
    assert np.isfinite(r["ate"])


def test_se_positive(data):
    r = gcomp(*data, n_boot=50)
    assert r["se"] >= 0


def test_ci_valid(data):
    r = gcomp(*data, n_boot=50)
    assert r["ci_lower"] <= r["ci_upper"]


def test_mean_treated_returned(data):
    r = gcomp(*data, n_boot=50)
    assert r["mean_treated"] is not None or r["mean_control"] is not None


def test_single_timepoint():
    rng = np.random.default_rng(11)
    n = 80
    T_seq = rng.binomial(1, 0.5, (n, 1)).astype(float)
    L_seq = rng.standard_normal((n, 1, 1))
    Y = 2.0 * T_seq[:, 0] + rng.standard_normal(n)
    r = gcomp(Y, T_seq, L_seq, n_boot=50)
    assert np.isfinite(r["ate"])


def test_cheatsheet():
    from morie.fn.gcomp import cheatsheet
    assert len(cheatsheet()) > 0
