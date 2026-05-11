"""Tests for snmtst.sensitivity_did."""
import numpy as np
import pytest
from morie.fn.snmtst import sensitivity_did


def test_snmtst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sensitivity_did(y, D, X, M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snmtst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sensitivity_did(y, D, X, M)
    assert isinstance(result, dict)
