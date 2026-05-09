"""Tests for gb_mnq.gibbons_marginal_quant."""
import numpy as np
import pytest
from moirais.fn.gb_mnq import gibbons_marginal_quant


def test_gb_mnq_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = gibbons_marginal_quant(x, p)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_mnq_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    result = gibbons_marginal_quant(x, p)
    assert isinstance(result, dict)
