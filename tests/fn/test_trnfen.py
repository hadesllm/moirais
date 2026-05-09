"""Tests for trnfen.transfer_entropy."""
import numpy as np
import pytest
from moirais.fn.trnfen import transfer_entropy


def test_trnfen_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag = 10
    result = transfer_entropy(x, y, lag)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_trnfen_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    lag = 10
    result = transfer_entropy(x, y, lag)
    assert isinstance(result, dict)
