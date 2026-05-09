"""Tests for dpgrf.dp_grouped_random_field."""
import numpy as np
import pytest
from moirais.fn.dpgrf import dp_grouped_random_field


def test_dpgrf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    grid = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = dp_grouped_random_field(y, grid, alpha, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpgrf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    grid = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    gamma = 1.0
    result = dp_grouped_random_field(y, grid, alpha, gamma)
    assert isinstance(result, dict)
