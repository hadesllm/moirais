"""Tests for netsts.neural_ts_lstm."""
import numpy as np
import pytest
from moirais.fn.netsts import neural_ts_lstm


def test_netsts_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    hidden = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = neural_ts_lstm(y, hidden, horizon)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_netsts_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    hidden = np.random.default_rng(42).normal(0, 1, 100)
    horizon = np.random.default_rng(42).normal(0, 1, 100)
    result = neural_ts_lstm(y, hidden, horizon)
    assert isinstance(result, dict)
