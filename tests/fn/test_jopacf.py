"""Tests for jopacf.joseph_partial_autocorrelation."""
import numpy as np
import pytest
from moirais.fn.jopacf import joseph_partial_autocorrelation


def test_jopacf_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_partial_autocorrelation(y, max_lag)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_jopacf_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_partial_autocorrelation(y, max_lag)
    assert isinstance(result, dict)
