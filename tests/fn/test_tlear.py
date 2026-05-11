"""Tests for morie.fn.tlear — T-learner CATE."""
import numpy as np
import pytest
from morie.fn.tlear import tlear


@pytest.fixture()
def data():
    rng = np.random.default_rng(1)
    n = 200
    x = rng.standard_normal((n, 2))
    t = rng.binomial(1, 0.5, n).astype(float)
    y = 1.5 + 2.5 * t + 0.4 * x[:, 0] + rng.standard_normal(n) * 0.3
    return y, t, x


def test_keys(data):
    r = tlear(*data)
    for k in ("cate", "ate", "se", "ci_lower", "ci_upper", "n", "method"):
        assert k in r


def test_ate_close_to_truth(data):
    r = tlear(*data)
    assert abs(r["ate"] - 2.5) < 0.6


def test_cate_shape(data):
    y, t, x = data
    assert tlear(*data)["cate"].shape == (len(y),)


def test_ci_valid(data):
    r = tlear(*data)
    assert r["ci_lower"] <= r["ate"] <= r["ci_upper"]


def test_method(data):
    assert tlear(*data)["method"] == "T-learner"


def test_separate_models_logic():
    """T-learner should differ from unadjusted mean difference."""
    rng = np.random.default_rng(7)
    n = 200
    x = rng.standard_normal((n, 1))
    t = (x[:, 0] > 0).astype(float)
    y = 1.0 + 3.0 * t - 1.5 * x[:, 0] + rng.standard_normal(n) * 0.2
    r = tlear(y, t, x)
    # Should be close to causal effect, not the naive diff
    naive = float(y[t == 1].mean() - y[t == 0].mean())
    assert abs(r["ate"] - 3.0) < abs(naive - 3.0) + 1.5


def test_logistic_model(data):
    y, t, x = data
    y_bin = (y > y.mean()).astype(float)
    r = tlear(y_bin, t, x, model="logistic")
    assert np.isfinite(r["ate"])


def test_cheatsheet():
    from morie.fn.tlear import cheatsheet
    assert len(cheatsheet()) > 0
