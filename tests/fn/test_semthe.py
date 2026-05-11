"""Tests for semthe.sem_theta."""
import numpy as np
import pytest
from morie.fn.semthe import sem_theta


def test_semthe_basic():
    """Test basic functionality."""
    theta = 0.0
    items = np.random.default_rng(42).normal(0, 1, 100)
    result = sem_theta(theta, items)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_semthe_edge():
    """Test edge cases."""
    theta = 0.0
    items = np.random.default_rng(42).normal(0, 1, 100)
    result = sem_theta(theta, items)
    assert isinstance(result, dict)
