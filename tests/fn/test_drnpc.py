"""Tests for drnpc.dr_did_neg_control."""
import numpy as np
import pytest
from morie.fn.drnpc import dr_did_neg_control


def test_drnpc_basic():
    """Test basic functionality."""
    y_main = np.random.default_rng(42).normal(0, 1, 100)
    y_neg = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_did_neg_control(y_main, y_neg, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drnpc_edge():
    """Test edge cases."""
    y_main = np.random.default_rng(42).normal(0, 1, 100)
    y_neg = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_did_neg_control(y_main, y_neg, D, X)
    assert isinstance(result, dict)
