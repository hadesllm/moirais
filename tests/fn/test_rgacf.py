"""Tests for rgacf.rangayyan_acf_estimate."""
import numpy as np
import pytest
from moirais.fn.rgacf import rangayyan_acf_estimate


def test_rgacf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_acf_estimate(x, max_lag)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgacf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_lag = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_acf_estimate(x, max_lag)
    assert isinstance(result, dict)
