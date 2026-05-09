"""Tests for joacf.joseph_autocorrelation_function."""
import numpy as np
import pytest
from moirais.fn.joacf import joseph_autocorrelation_function


def test_joacf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_autocorrelation_function(y, max_lag)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_joacf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_autocorrelation_function(y, max_lag)
    assert isinstance(result, dict)
