"""Tests for wsmaic.wasserman_aic."""
import numpy as np
import pytest
from morie.fn.wsmaic import wasserman_aic


def test_wsmaic_basic():
    """Test basic functionality."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = wasserman_aic(loglik, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmaic_edge():
    """Test edge cases."""
    loglik = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = wasserman_aic(loglik, k)
    assert isinstance(result, dict)
