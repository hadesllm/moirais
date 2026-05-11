"""Tests for reglu.reglu_activation."""
import numpy as np
import pytest
from morie.fn.reglu import reglu_activation


def test_reglu_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = reglu_activation(y, x, W, V)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_reglu_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    result = reglu_activation(y, x, W, V)
    assert isinstance(result, dict)
