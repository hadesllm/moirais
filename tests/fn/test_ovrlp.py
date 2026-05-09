"""Tests for moirais.fn.ovrlp — Overlap weights (causal inference) + Szymkiewicz-Simpson."""
import numpy as np
import pytest
from moirais.fn.ovrlp import ovrlp, overlap_coefficient


@pytest.fixture()
def causal_data():
    rng = np.random.default_rng(16)
    n = 300
    x = rng.standard_normal((n, 2))
    ps = 1 / (1 + np.exp(-(0.4 * x[:, 0])))
    t = rng.binomial(1, ps).astype(float)
    y = 2.0 * t + 0.3 * x[:, 0] + rng.standard_normal(n) * 0.4
    return y, t, x


def test_keys(causal_data):
    r = ovrlp(*causal_data)
    for k in ("ate", "se", "ci_lower", "ci_upper", "weights",
              "propensity", "effective_n", "n", "method"):
        assert k in r


def test_ate_close_to_truth(causal_data):
    r = ovrlp(*causal_data)
    assert abs(r["ate"] - 2.0) < 0.7


def test_weights_positive(causal_data):
    r = ovrlp(*causal_data)
    assert np.all(r["weights"] > 0)


def test_method(causal_data):
    assert ovrlp(*causal_data)["method"] == "overlap-weights"


def test_effective_n(causal_data):
    r = ovrlp(*causal_data)
    assert 0 < r["effective_n"] <= r["n"]


def test_ci_valid(causal_data):
    r = ovrlp(*causal_data)
    assert r["ci_lower"] <= r["ate"] <= r["ci_upper"]


def test_set_overlap_coefficient():
    oc = overlap_coefficient({1, 2, 3, 4}, {3, 4, 5, 6})
    assert abs(oc - 0.5) < 1e-9     # intersection={3,4}, min_size=4


def test_set_overlap_subset():
    oc = overlap_coefficient({1, 2}, {1, 2, 3, 4})
    assert abs(oc - 1.0) < 1e-9     # smaller set fully contained


def test_cheatsheet():
    from moirais.fn.ovrlp import cheatsheet
    assert len(cheatsheet()) > 0
