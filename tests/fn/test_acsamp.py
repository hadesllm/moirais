"""Tests for acsamp.sample_autocorrelation."""
import numpy as np
import pytest
from morie.fn.acsamp import sample_autocorrelation


def test_acsamp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_autocorrelation(y, max_lag)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_acsamp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_autocorrelation(y, max_lag)
    assert isinstance(result, dict)
