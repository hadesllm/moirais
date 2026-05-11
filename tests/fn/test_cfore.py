"""Tests for morie.fn.cfore — Causal forest CATE."""
import numpy as np
import pytest
from morie.fn.cfore import cfore


@pytest.fixture()
def data():
    rng = np.random.default_rng(4)
    n = 300
    x = rng.standard_normal((n, 2))
    t = rng.binomial(1, 0.5, n).astype(float)
    tau_true = 1.0 + 0.5 * x[:, 0]
    y = tau_true * t + 0.5 * x[:, 1] + rng.standard_normal(n) * 0.5
    return y, t, x


def test_keys(data):
    r = cfore(*data, n_trees=20, seed=0)
    for k in ("cate", "ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert k in r


def test_cate_shape(data):
    y, t, x = data
    r = cfore(*data, n_trees=20, seed=0)
    assert r["cate"].shape == (len(y),)


def test_ci_shape(data):
    r = cfore(*data, n_trees=20, seed=0)
    assert r["ci_lower"].shape == r["cate"].shape


def test_ate_finite(data):
    r = cfore(*data, n_trees=20, seed=0)
    assert np.isfinite(r["ate"])


def test_method(data):
    r = cfore(*data, n_trees=10, seed=42)
    assert r["method"] == "causal-forest"


def test_reproducible(data):
    y, t, x = data
    r1 = cfore(y, t, x, n_trees=10, seed=7)
    r2 = cfore(y, t, x, n_trees=10, seed=7)
    np.testing.assert_array_equal(r1["cate"], r2["cate"])


def test_heterogeneity_detected(data):
    """CATE should have non-zero variance (heterogeneous effects present)."""
    r = cfore(*data, n_trees=30, seed=0)
    assert float(np.std(r["cate"])) > 0.01


def test_cheatsheet():
    from morie.fn.cfore import cheatsheet
    assert len(cheatsheet()) > 0
