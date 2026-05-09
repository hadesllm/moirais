"""Tests for moirais.fn.elci — empirical likelihood CI."""

import numpy as np
import pytest

from moirais.fn.elci import elci


def test_ci_contains_mean():
    rng = np.random.default_rng(42)
    x = rng.standard_normal(300)
    result = elci(x)
    assert result["ci_lower"] < 0.0 < result["ci_upper"]


def test_ci_reasonable_width():
    rng = np.random.default_rng(7)
    x = rng.standard_normal(500) + 2.0
    result = elci(x)
    width = result["ci_upper"] - result["ci_lower"]
    assert 0 < width < 2.0


def test_empty_raises():
    with pytest.raises(ValueError, match="non-empty"):
        elci(np.array([]))
