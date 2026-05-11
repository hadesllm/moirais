"""Tests for b101.burkov_lm_ch1_linear_function."""
import numpy as np
import pytest
from morie.fn.b101 import burkov_lm_ch1_linear_function


def test_b101_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_linear_function(x, w, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b101_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_linear_function(x, w, b)
    assert isinstance(result, dict)
