"""Tests for grn021.geron_ch4_softmax_function."""
import numpy as np
import pytest
from morie.fn.grn021 import geron_ch4_softmax_function


def test_grn021_basic():
    """Test basic functionality."""
    s = 90
    k = 5
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = geron_ch4_softmax_function(s, k, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grn021_edge():
    """Test edge cases."""
    s = 90
    k = 5
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = geron_ch4_softmax_function(s, k, K)
    assert isinstance(result, dict)
