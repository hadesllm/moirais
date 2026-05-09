"""Tests for acfP.autocorrelation."""
import numpy as np
import pytest
from moirais.fn.acfP import autocorrelation


def test_acfP_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag_max = 100
    result = autocorrelation(y, lag_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_acfP_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag_max = 100
    result = autocorrelation(y, lag_max)
    assert isinstance(result, dict)
