"""Tests for vecmF.vecm."""
import numpy as np
import pytest
from morie.fn.vecmf import vecm


def test_vecmf_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    k_ar = np.random.default_rng(42).normal(0, 1, 100)
    coint_rank = np.random.default_rng(42).normal(0, 1, 100)
    result = vecm(Y, k_ar, coint_rank)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vecmf_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    k_ar = np.random.default_rng(42).normal(0, 1, 100)
    coint_rank = np.random.default_rng(42).normal(0, 1, 100)
    result = vecm(Y, k_ar, coint_rank)
    assert isinstance(result, dict)
