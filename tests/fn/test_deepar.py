"""Tests for deepar.deepar."""
import numpy as np
import pytest
from morie.fn.deepar import deepar


def test_deepar_basic():
    """Test basic functionality."""
    series = np.random.default_rng(42).normal(0, 1, 100)
    cov = np.random.default_rng(42).normal(0, 1, 100)
    lstm_h = np.random.default_rng(42).normal(0, 1, 100)
    result = deepar(series, cov, lstm_h)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deepar_edge():
    """Test edge cases."""
    series = np.random.default_rng(42).normal(0, 1, 100)
    cov = np.random.default_rng(42).normal(0, 1, 100)
    lstm_h = np.random.default_rng(42).normal(0, 1, 100)
    result = deepar(series, cov, lstm_h)
    assert isinstance(result, dict)
