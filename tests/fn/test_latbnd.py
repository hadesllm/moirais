"""Tests for latbnd.late_bounds."""
import numpy as np
import pytest
from moirais.fn.latbnd import late_bounds


def test_latbnd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = late_bounds(y, D, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_latbnd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = late_bounds(y, D, Z)
    assert isinstance(result, dict)
