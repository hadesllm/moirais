"""Tests for morie.fn.adkde — Adaptive KDE."""

import numpy as np
import pytest
from morie.fn.adkde import adkde


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = adkde(x)
    assert isinstance(result, dict)
    for key in ("x_eval", "density", "local_bandwidths", "bandwidth", "alpha", "n_obs"):
        assert key in result


def test_density_nonnegative():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = adkde(x)
    assert all(d >= -1e-10 for d in result["density"])


def test_alpha_validation():
    with pytest.raises(ValueError, match="alpha"):
        adkde(np.ones(10), alpha=1.5)


def test_variable_bandwidths():
    rng = np.random.default_rng(42)
    x = np.concatenate([rng.normal(0, 0.5, 50), rng.normal(5, 2.0, 50)])
    result = adkde(x)
    bws = result["local_bandwidths"]
    assert len(set(round(b, 6) for b in bws)) > 1
