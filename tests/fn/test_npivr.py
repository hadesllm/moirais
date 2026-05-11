"""Tests for morie.fn.npivr — Nonparametric instrumental variables."""

import numpy as np
import pytest

from morie.fn.npivr import npivr


@pytest.fixture()
def iv_data():
    rng = np.random.default_rng(42)
    n = 300
    z = rng.standard_normal(n)
    u = rng.standard_normal(n)
    d = 0.8 * z + 0.3 * u + rng.normal(0, 0.2, n)
    y = 1.5 * d - 0.3 * u + rng.normal(0, 0.3, n)
    return y, d, z


def test_returns_dict(iv_data):
    y, d, z = iv_data
    result = npivr(y, d, z)
    assert isinstance(result, dict)
    for key in ("d_hat", "g_hat", "avg_deriv", "se_avg_deriv", "bandwidth_dz", "bandwidth_z", "n_obs"):
        assert key in result


def test_avg_deriv_positive(iv_data):
    y, d, z = iv_data
    result = npivr(y, d, z)
    assert result["avg_deriv"] > 0


def test_se_positive(iv_data):
    y, d, z = iv_data
    result = npivr(y, d, z)
    assert result["se_avg_deriv"] > 0


def test_d_hat_length(iv_data):
    y, d, z = iv_data
    result = npivr(y, d, z)
    assert len(result["d_hat"]) == len(y)


def test_g_hat_finite(iv_data):
    y, d, z = iv_data
    result = npivr(y, d, z)
    assert np.all(np.isfinite(result["g_hat"]))


def test_with_covariates(iv_data):
    y, d, z = iv_data
    rng = np.random.default_rng(99)
    x = rng.standard_normal(len(y))
    result = npivr(y, d, z, x)
    assert result["n_obs"] == len(y)


def test_mismatched_raises():
    with pytest.raises(ValueError, match="same length"):
        npivr(np.ones(10), np.ones(5), np.ones(10))


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 10"):
        npivr(np.ones(5), np.ones(5), np.ones(5))
