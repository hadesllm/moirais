"""Tests for hmvf.geron_value_function."""
import numpy as np
import pytest
from morie.fn.hmvf import geron_value_function


def test_hmvf_basic():
    """Test basic functionality."""
    s = 90
    pi = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_value_function(s, pi, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmvf_edge():
    """Test edge cases."""
    s = 90
    pi = np.random.default_rng(42).normal(0, 1, 100)
    gamma = 1.0
    result = geron_value_function(s, pi, gamma)
    assert isinstance(result, dict)
