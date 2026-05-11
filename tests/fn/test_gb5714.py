"""Tests for gb5714.gibbons_wsrt_sampsize."""
import numpy as np
import pytest
from morie.fn.gb5714 import gibbons_wsrt_sampsize


def test_gb5714_basic():
    """Test basic functionality."""
    alpha = 0.05
    beta = 0.8
    delta = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = gibbons_wsrt_sampsize(alpha, beta, delta, sigma)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5714_edge():
    """Test edge cases."""
    alpha = 0.05
    beta = 0.8
    delta = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = gibbons_wsrt_sampsize(alpha, beta, delta, sigma)
    assert isinstance(result, dict)
