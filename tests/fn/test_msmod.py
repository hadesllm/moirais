"""Tests for moirais.fn.msmod — Marginal structural model."""
import numpy as np
import pytest
from moirais.fn.msmod import msmod


@pytest.fixture()
def data():
    rng = np.random.default_rng(5)
    n = 300
    x = rng.standard_normal((n, 2))
    ps = 1 / (1 + np.exp(-(0.3 * x[:, 0])))
    t = rng.binomial(1, ps).astype(float)
    y = 2.0 * t + 0.5 * x[:, 0] + rng.standard_normal(n) * 0.5
    return y, t, x


def test_keys(data):
    r = msmod(*data)
    for k in ("ate", "se", "ci_lower", "ci_upper", "weights", "effective_n", "n", "method"):
        assert k in r


def test_ate_close_to_truth(data):
    r = msmod(*data)
    assert abs(r["ate"] - 2.0) < 0.6


def test_weights_positive(data):
    r = msmod(*data)
    assert np.all(r["weights"] > 0)


def test_effective_n_leq_n(data):
    r = msmod(*data)
    assert r["effective_n"] <= r["n"]


def test_stabilize_false(data):
    y, t, x = data
    r = msmod(y, t, x, stabilize=False)
    assert np.isfinite(r["ate"])


def test_method(data):
    assert msmod(*data)["method"] == "MSM-IPW"


def test_ci_valid(data):
    r = msmod(*data)
    assert r["ci_lower"] <= r["ate"] <= r["ci_upper"]


def test_cheatsheet():
    from moirais.fn.msmod import cheatsheet
    assert len(cheatsheet()) > 0
