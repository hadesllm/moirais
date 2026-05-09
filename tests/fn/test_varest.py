"""Tests for varest.vector_autoregression."""
import numpy as np
import pytest
from moirais.fn.varest import vector_autoregression


def test_varest_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = vector_autoregression(Y, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_varest_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = vector_autoregression(Y, p)
    assert isinstance(result, dict)
