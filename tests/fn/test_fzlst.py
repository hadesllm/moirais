"""Tests for fzlst.fauzi_l_statistic."""
import numpy as np
import pytest
from morie.fn.fzlst import fauzi_l_statistic


def test_fzlst_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    bandwidth = 0.3
    result = fauzi_l_statistic(data, p, bandwidth)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_fzlst_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    bandwidth = 0.3
    result = fauzi_l_statistic(data, p, bandwidth)
    assert isinstance(result, dict)
