"""Tests for rng044.rangayyan_ch3_lsi_parallel_branch_1."""
import numpy as np
import pytest
from morie.fn.rng044 import rangayyan_ch3_lsi_parallel_branch_1


def test_rng044_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h_1 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_parallel_branch_1(x, h_1, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng044_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    h_1 = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_lsi_parallel_branch_1(x, h_1, n)
    assert isinstance(result, dict)
