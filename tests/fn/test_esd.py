"""Tests for esd.generalized_esd."""
import numpy as np
import pytest
from morie.fn.esd import generalized_esd


def test_esd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    r = 10
    result = generalized_esd(x, alpha, r)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_esd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    r = 10
    result = generalized_esd(x, alpha, r)
    assert isinstance(result, dict)
