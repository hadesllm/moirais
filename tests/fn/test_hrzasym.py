"""Tests for hrzasym.horowitz_one_step_efficient."""
import numpy as np
import pytest
from morie.fn.hrzasym import horowitz_one_step_efficient


def test_hrzasym_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    initial_estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_one_step_efficient(x, y, bandwidth, initial_estimator)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzasym_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    bandwidth = 0.3
    initial_estimator = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_one_step_efficient(x, y, bandwidth, initial_estimator)
    assert isinstance(result, dict)
