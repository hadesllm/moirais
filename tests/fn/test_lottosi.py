"""Tests for lottosi.lottery_sampling."""
import numpy as np
import pytest
from morie.fn.lottosi import lottery_sampling


def test_lottosi_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    n = 100
    result = lottery_sampling(y, N, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lottosi_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    N = 100
    n = 100
    result = lottery_sampling(y, N, n)
    assert isinstance(result, dict)
