"""Tests for morie.fn.waic -- WAIC."""

import numpy as np
from morie.fn.waic import compute_waic


def test_returns_dict():
    rng = np.random.default_rng(42)
    ll = rng.normal(0, 1, (100, 20))
    result = compute_waic(ll)
    assert isinstance(result, dict)
    assert "waic" in result
    assert "lppd" in result
    assert "p_waic" in result


def test_pointwise_shape():
    rng = np.random.default_rng(42)
    ll = rng.normal(0, 1, (50, 10))
    result = compute_waic(ll)
    assert len(result["pointwise_waic"]) == 10


def test_waic_equals_sum():
    rng = np.random.default_rng(42)
    ll = rng.normal(0, 1, (50, 10))
    result = compute_waic(ll)
    np.testing.assert_allclose(
        result["waic"],
        -2 * (result["lppd"] - result["p_waic"]),
        atol=1e-8,
    )


def test_p_waic_nonnegative():
    rng = np.random.default_rng(42)
    ll = rng.normal(0, 1, (50, 10))
    result = compute_waic(ll)
    assert result["p_waic"] >= 0


def test_invalid_1d():
    try:
        compute_waic([1.0, 2.0, 3.0])
        assert False
    except ValueError:
        pass


def test_too_few_samples():
    try:
        compute_waic([[1.0, 2.0]])
        assert False
    except ValueError:
        pass
