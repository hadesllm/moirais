"""Tests for moirais.fn.brent — bracketing entropy."""

import numpy as np
import pytest

from moirais.fn.brent import brent


def test_basic_output():
    rng = np.random.default_rng(42)
    data = rng.standard_normal((20, 50))
    result = brent(data)
    assert "bracketing_numbers" in result
    assert "entropy_integral" in result
    assert result["entropy_integral"] >= 0


def test_identical_functions():
    data = np.ones((5, 10))
    result = brent(data)
    assert np.all(result["bracketing_numbers"] <= 1)


def test_l2_vs_sup():
    rng = np.random.default_rng(7)
    data = rng.standard_normal((10, 30))
    r_l2 = brent(data, metric="l2")
    r_sup = brent(data, metric="sup")
    assert r_l2["entropy_integral"] != r_sup["entropy_integral"]


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        brent(np.array([]))
