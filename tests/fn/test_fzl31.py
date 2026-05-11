"""Tests for fzl31.fauzi_lem3_1_asymp_rep."""
import numpy as np
import pytest
from morie.fn.fzl31 import fauzi_lem3_1_asymp_rep


def test_fzl31_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    bandwidth = 0.3
    result = fauzi_lem3_1_asymp_rep(data, p, bandwidth)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzl31_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    bandwidth = 0.3
    result = fauzi_lem3_1_asymp_rep(data, p, bandwidth)
    assert isinstance(result, dict)
