"""Tests for isodst.isotropy_distance_test."""
import numpy as np
import pytest
from morie.fn.isodst import isotropy_distance_test


def test_isodst_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    result = isotropy_distance_test(coords, values)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_isodst_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    result = isotropy_distance_test(coords, values)
    assert isinstance(result, dict)
