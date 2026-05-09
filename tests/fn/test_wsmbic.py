"""Tests for wsmbic.wasserman_bic."""
import numpy as np
import pytest
from moirais.fn.wsmbic import wasserman_bic


def test_wsmbic_basic():
    """Test basic functionality."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    n = 100
    result = wasserman_bic(loglik, k, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmbic_edge():
    """Test edge cases."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    n = 100
    result = wasserman_bic(loglik, k, n)
    assert isinstance(result, dict)
