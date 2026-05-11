"""Tests for hot.hot_sax."""
import numpy as np
import pytest
from morie.fn.hot import hot_sax


def test_hot_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    alphabet = np.random.default_rng(42).normal(0, 1, 100)
    result = hot_sax(x, window, alphabet)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hot_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    alphabet = np.random.default_rng(42).normal(0, 1, 100)
    result = hot_sax(x, window, alphabet)
    assert isinstance(result, dict)
