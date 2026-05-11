"""Tests for morie.fn.looic -- LOO-IC."""

import numpy as np
from morie.fn.looic import compute_loo


def test_returns_dict():
    rng = np.random.default_rng(42)
    ll = rng.normal(-1, 0.3, (100, 20))
    result = compute_loo(ll)
    assert isinstance(result, dict)
    assert "loo" in result
    assert "looic" in result
    assert "pareto_k" in result


def test_looic_is_neg2_loo():
    rng = np.random.default_rng(42)
    ll = rng.normal(-1, 0.3, (100, 20))
    result = compute_loo(ll)
    np.testing.assert_allclose(result["looic"], -2 * result["loo"], atol=1e-8)


def test_pareto_k_shape():
    rng = np.random.default_rng(42)
    ll = rng.normal(-1, 0.3, (50, 15))
    result = compute_loo(ll)
    assert len(result["pareto_k"]) == 15


def test_se_positive():
    rng = np.random.default_rng(42)
    ll = rng.normal(-1, 0.3, (100, 20))
    result = compute_loo(ll)
    assert result["se"] > 0


def test_invalid_1d():
    try:
        compute_loo([1.0, 2.0, 3.0])
        assert False
    except ValueError:
        pass
