"""Tests for moirais.fn.bloos -- PSIS-LOO."""

import numpy as np
from moirais.fn.bloos import psis_loo


def test_returns_dict():
    ll = -0.5 * np.random.default_rng(42).standard_normal((50, 20)) ** 2
    result = psis_loo(ll)
    assert isinstance(result, dict)
    assert "elpd_loo" in result


def test_looic_finite():
    ll = -0.5 * np.random.default_rng(42).standard_normal((100, 30)) ** 2
    result = psis_loo(ll)
    assert np.isfinite(result["looic"])


def test_k_hat_length():
    ll = -0.5 * np.random.default_rng(42).standard_normal((50, 10)) ** 2
    result = psis_loo(ll)
    assert len(result["k_hat"]) == 10
