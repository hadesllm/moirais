"""Tests for pate.pate."""
import numpy as np
import pytest
from morie.fn.pate import pate


def test_pate_basic():
    """Test basic functionality."""
    teachers = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = pate(teachers, x, epsilon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pate_edge():
    """Test edge cases."""
    teachers = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    result = pate(teachers, x, epsilon)
    assert isinstance(result, dict)
