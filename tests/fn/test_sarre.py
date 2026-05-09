"""Tests for sarre.spatial_ar_error."""
import numpy as np
import pytest
from moirais.fn.sarre import spatial_ar_error


def test_sarre_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = spatial_ar_error(x, y, w)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sarre_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    result = spatial_ar_error(x, y, w)
    assert isinstance(result, dict)
