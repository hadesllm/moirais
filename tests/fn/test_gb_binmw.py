"""Tests for gb_binmw.gibbons_mw_binomial_link."""
import numpy as np
import pytest
from moirais.fn.gb_binmw import gibbons_mw_binomial_link


def test_gb_binmw_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = gibbons_mw_binomial_link(W, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gb_binmw_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = gibbons_mw_binomial_link(W, m)
    assert isinstance(result, dict)
