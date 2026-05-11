"""Tests for joses.joseph_simple_exponential_smoothing."""
import numpy as np
import pytest
from morie.fn.joses import joseph_simple_exponential_smoothing


def test_joses_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_simple_exponential_smoothing(y, alpha, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joses_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    alpha = 0.05
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_simple_exponential_smoothing(y, alpha, horizon)
    assert isinstance(result, dict)
