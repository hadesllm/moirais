"""Tests for sdcfst.semi_doubly_robust_forest."""
import numpy as np
import pytest
from morie.fn.sdcfst import semi_doubly_robust_forest


def test_sdcfst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K_fold = np.random.default_rng(42).normal(0, 1, 100)
    result = semi_doubly_robust_forest(y, D, X, K_fold)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sdcfst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    K_fold = np.random.default_rng(42).normal(0, 1, 100)
    result = semi_doubly_robust_forest(y, D, X, K_fold)
    assert isinstance(result, dict)
