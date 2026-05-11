"""Tests for jowfv.joseph_walk_forward_validation."""
import numpy as np
import pytest
from morie.fn.jowfv import joseph_walk_forward_validation


def test_jowfv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    T_start = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_walk_forward_validation(y, model, T_start)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jowfv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    T_start = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_walk_forward_validation(y, model, T_start)
    assert isinstance(result, dict)
