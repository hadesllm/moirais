"""Tests for lggvls.laggedval_iptw."""
import numpy as np
import pytest
from moirais.fn.lggvls import laggedval_iptw


def test_lggvls_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    lag = 10
    result = laggedval_iptw(y, A, H, lag)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lggvls_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    H = np.random.default_rng(42).normal(0, 1, 100)
    lag = 10
    result = laggedval_iptw(y, A, H, lag)
    assert isinstance(result, dict)
