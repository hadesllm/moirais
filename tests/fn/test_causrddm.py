"""Tests for causrddm.causal_rdd_manipulation."""
import numpy as np
import pytest
from morie.fn.causrddm import causal_rdd_manipulation


def test_causrddm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    bw = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_rdd_manipulation(x, cutoff, bw)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_causrddm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    cutoff = 10.0
    bw = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_rdd_manipulation(x, cutoff, bw)
    assert isinstance(result, dict)
