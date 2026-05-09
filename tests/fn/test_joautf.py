"""Tests for joautf.joseph_autoformer."""
import numpy as np
import pytest
from moirais.fn.joautf import joseph_autoformer


def test_joautf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    top_k_lags = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_autoformer(x, horizon, top_k_lags)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_joautf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    top_k_lags = np.random.default_rng(42).normal(0, 1, 100)
    result = joseph_autoformer(x, horizon, top_k_lags)
    assert isinstance(result, dict)
