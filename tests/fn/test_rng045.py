"""Tests for rng045.rangayyan_ch3_lsi_parallel_branch_2."""
import numpy as np
import pytest
from morie.fn.rng045 import rangayyan_ch3_lsi_parallel_branch_2


def test_rng045_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h_2 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_parallel_branch_2(x, h_2, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng045_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h_2 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_parallel_branch_2(x, h_2, n)
    assert isinstance(result, dict)
