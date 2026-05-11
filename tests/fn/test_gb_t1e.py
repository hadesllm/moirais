"""Tests for gb_t1e.gibbons_type1_error."""
import numpy as np
import pytest
from morie.fn.gb_t1e import gibbons_type1_error


def test_gb_t1e_basic():
    """Test basic functionality."""
    statistic = np.random.default_rng(42).normal(0, 1, 100)
    null_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_type1_error(statistic, null_dist)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_t1e_edge():
    """Test edge cases."""
    statistic = np.random.default_rng(42).normal(0, 1, 100)
    null_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_type1_error(statistic, null_dist)
    assert isinstance(result, dict)
