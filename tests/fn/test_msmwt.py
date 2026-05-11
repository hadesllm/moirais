"""Tests for morie.fn.msmwt — MSM stabilized weights."""
import numpy as np
import pytest
from morie.fn.msmwt import msmwt


@pytest.fixture()
def data():
    rng = np.random.default_rng(17)
    n, K = 150, 3
    T_seq = rng.binomial(1, 0.5, (n, K)).astype(float)
    L_seq = rng.standard_normal((n, K))
    return T_seq, L_seq


def test_keys(data):
    r = msmwt(*data)
    for k in ("weights", "weight_numerator", "weight_denominator",
              "effective_n", "max_weight", "mean_weight", "n", "K", "method"):
        assert k in r


def test_weight_shape(data):
    T_seq, L_seq = data
    r = msmwt(*data)
    assert r["weights"].shape == (T_seq.shape[0],)


def test_weights_positive(data):
    r = msmwt(*data)
    assert np.all(r["weights"] > 0)


def test_effective_n_leq_n(data):
    r = msmwt(*data)
    assert r["effective_n"] <= r["n"]


def test_stabilized_less_extreme(data):
    r_stab = msmwt(*data, stabilize=True)
    r_raw = msmwt(*data, stabilize=False)
    # Stabilized weights should have smaller max
    assert r_stab["max_weight"] <= r_raw["max_weight"] + 1.0


def test_method_label(data):
    r = msmwt(*data)
    assert "MSM" in r["method"]


def test_K_correct(data):
    T_seq, _ = data
    r = msmwt(*data)
    assert r["K"] == T_seq.shape[1]


def test_cheatsheet():
    from morie.fn.msmwt import cheatsheet
    assert len(cheatsheet()) > 0
