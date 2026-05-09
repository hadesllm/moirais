"""Tests for moirais.fn.bwaic -- WAIC."""

import numpy as np
from moirais.fn.bwaic import compute_waic


def test_returns_dict():
    ll = np.random.default_rng(42).standard_normal((50, 20))
    result = compute_waic(ll)
    assert isinstance(result, dict)
    assert "waic" in result


def test_waic_finite():
    ll = -0.5 * np.random.default_rng(42).standard_normal((50, 20)) ** 2
    result = compute_waic(ll)
    assert np.isfinite(result["waic"])


def test_p_waic_non_negative():
    ll = -0.5 * np.random.default_rng(42).standard_normal((100, 30)) ** 2
    result = compute_waic(ll)
    assert result["p_waic"] >= 0
