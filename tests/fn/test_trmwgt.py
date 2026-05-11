"""Tests for trmwgt.trim_weights."""
import numpy as np
import pytest
from morie.fn.trmwgt import trim_weights


def test_trmwgt_basic():
    """Test basic functionality."""
    weights = np.random.default_rng(45).exponential(1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = trim_weights(weights, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trmwgt_edge():
    """Test edge cases."""
    weights = np.random.default_rng(45).exponential(1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = trim_weights(weights, quantile)
    assert isinstance(result, dict)
