"""Tests for causrddh.causal_rdd_imbens_kalyanaraman."""
import numpy as np
import pytest
from morie.fn.causrddh import causal_rdd_imbens_kalyanaraman


def test_causrddh_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    cutoff = 10.0
    result = causal_rdd_imbens_kalyanaraman(x, y, cutoff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causrddh_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    cutoff = 10.0
    result = causal_rdd_imbens_kalyanaraman(x, y, cutoff)
    assert isinstance(result, dict)
