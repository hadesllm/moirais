"""Tests for causinst.causal_iv_instrumental_dag."""
import numpy as np
import pytest
from morie.fn.causinst import causal_iv_instrumental_dag


def test_causinst_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_iv_instrumental_dag(y, D, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causinst_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = causal_iv_instrumental_dag(y, D, Z)
    assert isinstance(result, dict)
