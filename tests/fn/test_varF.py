"""Tests for varF.vector_autoregression."""
import numpy as np
import pytest
from morie.fn.varF import vector_autoregression


def test_varF_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = vector_autoregression(Y, p)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_varF_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    result = vector_autoregression(Y, p)
    assert isinstance(result, dict)
