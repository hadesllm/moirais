"""Tests for hmmxp.geron_max_pool."""
import numpy as np
import pytest
from morie.fn.hmmxp import geron_max_pool


def test_hmmxp_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_max_pool(x, window, stride)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmmxp_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    stride = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_max_pool(x, window, stride)
    assert isinstance(result, dict)
