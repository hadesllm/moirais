"""Tests for morie.fn.drlea — DR-learner CATE."""
import numpy as np
import pytest
from morie.fn.drlea import drlea


@pytest.fixture()
def data():
    rng = np.random.default_rng(3)
    n = 200
    x = rng.standard_normal((n, 2))
    ps = 1 / (1 + np.exp(-(0.4 * x[:, 0])))
    t = rng.binomial(1, ps).astype(float)
    y = 1.0 + 2.0 * t + 0.4 * x[:, 0] + rng.standard_normal(n) * 0.3
    return y, t, x


def test_keys(data):
    r = drlea(*data)
    for k in ("cate", "ate", "se", "ci_lower", "ci_upper", "pseudo_outcome", "n", "method"):
        assert k in r


def test_ate_close_to_truth(data):
    r = drlea(*data)
    assert abs(r["ate"] - 2.0) < 0.5


def test_pseudo_outcome_shape(data):
    y, t, x = data
    r = drlea(*data)
    assert r["pseudo_outcome"].shape == (len(y),)


def test_doubly_robust_property():
    """DR estimator should be consistent even with misspecified PS."""
    rng = np.random.default_rng(10)
    n = 300
    x = rng.standard_normal((n, 1))
    t = rng.binomial(1, 0.5, n).astype(float)  # balanced
    y = 3.0 * t + rng.standard_normal(n) * 0.3
    r = drlea(y, t, x)
    assert abs(r["ate"] - 3.0) < 0.6


def test_se_positive(data):
    assert drlea(*data)["se"] > 0


def test_ci_valid(data):
    r = drlea(*data)
    assert r["ci_lower"] <= r["ate"] <= r["ci_upper"]


def test_method(data):
    assert drlea(*data)["method"] == "DR-learner"


def test_cheatsheet():
    from morie.fn.drlea import cheatsheet
    assert len(cheatsheet()) > 0
