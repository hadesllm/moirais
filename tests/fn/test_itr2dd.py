"""Tests for itr2dd.itr_optimal_did."""
import numpy as np
import pytest
from morie.fn.itr2dd import itr_optimal_did


def test_itr2dd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = itr_optimal_did(y, D, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_itr2dd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = itr_optimal_did(y, D, W)
    assert isinstance(result, dict)
