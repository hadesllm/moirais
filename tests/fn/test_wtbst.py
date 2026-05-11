"""Tests for morie.fn.wtbst — weighted bootstrap."""

import numpy as np
import pytest

from morie.fn.wtbst import wtbst


def test_basic_output():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(200)
    result = wtbst(x, n_boot=300, seed=7)
    assert "estimate" in result
    assert "se" in result
    assert result["n"] == 200


def test_ci_contains_mean():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(500) + 2.0
    result = wtbst(x, n_boot=500, seed=1)
    assert result["ci_lower"] < 2.0 < result["ci_upper"]


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        wtbst(np.array([]))
