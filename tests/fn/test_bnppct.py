"""Tests for bnppct.bnp_percent_quantile."""
import numpy as np
import pytest
from moirais.fn.bnppct import bnp_percent_quantile


def test_bnppct_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = bnp_percent_quantile(y, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bnppct_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = bnp_percent_quantile(y, quantile)
    assert isinstance(result, dict)
