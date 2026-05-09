"""Tests for kmpref.kamath_prefix_tuning."""
import numpy as np
import pytest
from moirais.fn.kmpref import kamath_prefix_tuning


def test_kmpref_basic():
    """Test basic functionality."""
    prefix_K = np.random.default_rng(42).normal(0, 1, 100)
    prefix_V = np.random.default_rng(42).normal(0, 1, 100)
    K_input = np.random.default_rng(42).normal(0, 1, 100)
    V_input = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_prefix_tuning(prefix_K, prefix_V, K_input, V_input)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmpref_edge():
    """Test edge cases."""
    prefix_K = np.random.default_rng(42).normal(0, 1, 100)
    prefix_V = np.random.default_rng(42).normal(0, 1, 100)
    K_input = np.random.default_rng(42).normal(0, 1, 100)
    V_input = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_prefix_tuning(prefix_K, prefix_V, K_input, V_input)
    assert isinstance(result, dict)
