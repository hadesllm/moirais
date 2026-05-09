"""Tests for npbqr.np_bayes_quant_reg."""
import numpy as np
import pytest
from moirais.fn.npbqr import np_bayes_quant_reg


def test_npbqr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tau = 0.1
    result = np_bayes_quant_reg(y, X, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_npbqr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    tau = 0.1
    result = np_bayes_quant_reg(y, X, tau)
    assert isinstance(result, dict)
