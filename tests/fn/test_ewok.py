"""Tests for moirais.fn.ewok — Glass's delta."""
import numpy as np
import pytest

from moirais.fn.ewok import glass_delta, ewok


def test_glass_delta_known_diff():
    """Two groups with known mean difference, control SD = 1."""
    rng = np.random.default_rng(42)
    y = rng.standard_normal(200)  # control
    x = rng.standard_normal(200) + 1.0  # treatment shifted by 1
    result = glass_delta(x, y)
    assert abs(result.estimate - 1.0) < 0.5, f"Glass delta={result.estimate}, expected ~1.0"
    assert result.se > 0
    assert result.ci_lower < result.estimate < result.ci_upper


def test_glass_delta_zero_diff():
    """Same distribution should give delta near zero."""
    rng = np.random.default_rng(42)
    x = rng.standard_normal(100)
    y = rng.standard_normal(100)
    result = glass_delta(x, y)
    assert abs(result.estimate) < 0.5


def test_ewok_alias():
    assert ewok is glass_delta


def test_glass_delta_small_n_raises():
    with pytest.raises(ValueError):
        glass_delta([1], [2, 3])
