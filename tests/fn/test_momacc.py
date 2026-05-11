"""Tests for momacc.moments_accountant."""
import numpy as np
import pytest
from morie.fn.momacc import moments_accountant


def test_momacc_basic():
    """Test basic functionality."""
    sigma = 1.0
    sample_rate = 0.1
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = moments_accountant(sigma, sample_rate, steps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_momacc_edge():
    """Test edge cases."""
    sigma = 1.0
    sample_rate = 0.1
    steps = np.random.default_rng(42).normal(0, 1, 100)
    result = moments_accountant(sigma, sample_rate, steps)
    assert isinstance(result, dict)
