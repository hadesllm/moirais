"""Tests for moirais.fn.bkde — Boundary-corrected KDE."""

import numpy as np
import pytest
from moirais.fn.bkde import bkde


def test_returns_dict():
    rng = np.random.default_rng(42)
    x = np.abs(rng.standard_normal(100))
    result = bkde(x, lower=0.0)
    assert isinstance(result, dict)
    for key in ("x_eval", "density", "bandwidth", "method", "n_obs"):
        assert key in result


def test_reflection_method():
    rng = np.random.default_rng(42)
    x = np.abs(rng.standard_normal(100))
    result = bkde(x, lower=0.0, method="reflection")
    assert result["method"] == "reflection"


def test_renormalization_method():
    rng = np.random.default_rng(42)
    x = np.abs(rng.standard_normal(100))
    result = bkde(x, lower=0.0, method="renormalization")
    assert result["method"] == "renormalization"


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        bkde(np.ones(10), method="invalid")
