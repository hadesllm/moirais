"""Tests for morie.fn.npqnt — Nonparametric quantile regression."""

import numpy as np
import pytest

from morie.fn.npqnt import npqnt


@pytest.fixture()
def het_data():
    rng = np.random.default_rng(42)
    n = 200
    x = rng.uniform(0, 10, n)
    y = 2 * x + x * rng.normal(0, 0.5, n)
    return x, y


def test_returns_dict(het_data):
    x, y = het_data
    result = npqnt(x, y)
    assert isinstance(result, dict)
    for key in ("x_eval", "quantiles", "tau", "bandwidth", "kernel", "n_obs"):
        assert key in result


def test_median_default(het_data):
    x, y = het_data
    result = npqnt(x, y)
    assert result["tau"] == 0.5


def test_quantiles_finite(het_data):
    x, y = het_data
    result = npqnt(x, y, x_eval=np.array([2.0, 5.0, 8.0]))
    assert np.all(np.isfinite(result["quantiles"]))


def test_upper_above_lower(het_data):
    x, y = het_data
    x_eval = np.array([3.0, 6.0])
    q25 = npqnt(x, y, tau=0.25, x_eval=x_eval)["quantiles"]
    q75 = npqnt(x, y, tau=0.75, x_eval=x_eval)["quantiles"]
    assert np.all(q75 >= q25 - 1.0)


def test_tau_out_of_range():
    with pytest.raises(ValueError, match="tau must be in"):
        npqnt(np.arange(10.0), np.arange(10.0), tau=1.5)


def test_tau_zero():
    with pytest.raises(ValueError, match="tau must be in"):
        npqnt(np.arange(10.0), np.arange(10.0), tau=0.0)


def test_mismatched_raises():
    with pytest.raises(ValueError, match="same length"):
        npqnt(np.ones(5), np.ones(3))


def test_too_few_raises():
    with pytest.raises(ValueError, match="at least 4"):
        npqnt(np.ones(3), np.ones(3))
