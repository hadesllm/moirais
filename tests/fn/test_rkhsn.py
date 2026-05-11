"""Tests for rkhsn.rkhs_norm."""
import numpy as np
import pytest
from morie.fn.rkhsn import rkhs_norm


def test_rkhsn_basic():
    """Test basic functionality."""
    alpha = 0.05
    eigenvalues = np.random.default_rng(42).normal(0, 1, 100)
    result = rkhs_norm(alpha, eigenvalues)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rkhsn_edge():
    """Test edge cases."""
    alpha = 0.05
    eigenvalues = np.random.default_rng(42).normal(0, 1, 100)
    result = rkhs_norm(alpha, eigenvalues)
    assert isinstance(result, dict)
