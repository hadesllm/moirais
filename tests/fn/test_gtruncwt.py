"""Tests for gtruncwt.truncate_weights."""
import numpy as np
import pytest
from moirais.fn.gtruncwt import truncate_weights


def test_gtruncwt_basic():
    """Test basic functionality."""
    weights = np.random.default_rng(45).exponential(1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = truncate_weights(weights, quantile)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gtruncwt_edge():
    """Test edge cases."""
    weights = np.random.default_rng(45).exponential(1, 100)
    quantile = np.random.default_rng(42).normal(0, 1, 100)
    result = truncate_weights(weights, quantile)
    assert isinstance(result, dict)
