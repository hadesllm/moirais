"""Tests for moirais.fn.bysbt — Bayesian bootstrap."""

import numpy as np
import pytest

from moirais.fn.bysbt import bysbt


def test_mean_close_to_sample():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200) + 3.0
    result = bysbt(x, n_boot=500, seed=7)
    assert result["estimate"] == pytest.approx(3.0, abs=0.3)


def test_ci_contains_mean():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(300) + 1.0
    result = bysbt(x, n_boot=500, seed=42)
    assert result["ci_lower"] < 1.0 < result["ci_upper"]


def test_se_positive():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    result = bysbt(x, n_boot=200, seed=1)
    assert result["se"] > 0


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        bysbt(np.array([]))
