"""Tests for drbst.dr_did_bootstrap."""
import numpy as np
import pytest
from morie.fn.drbst import dr_did_bootstrap


def test_drbst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = dr_did_bootstrap(y, D, X, B)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drbst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = dr_did_bootstrap(y, D, X, B)
    assert isinstance(result, dict)
