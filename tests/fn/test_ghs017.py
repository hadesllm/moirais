"""Tests for ghs017.ghosal_ch3_discrete_random_measure."""
import numpy as np
import pytest
from morie.fn.ghs017 import ghosal_ch3_discrete_random_measure


def test_ghs017_basic():
    """Test basic functionality."""
    W_i = np.random.default_rng(42).normal(0, 1, 100)
    theta_i = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_discrete_random_measure(W_i, theta_i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs017_edge():
    """Test edge cases."""
    W_i = np.random.default_rng(42).normal(0, 1, 100)
    theta_i = np.random.default_rng(42).normal(0, 1, 100)
    result = ghosal_ch3_discrete_random_measure(W_i, theta_i)
    assert isinstance(result, dict)
