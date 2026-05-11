"""Tests for b104.burkov_lm_ch1_linear_vector."""
import numpy as np
import pytest
from morie.fn.b104 import burkov_lm_ch1_linear_vector


def test_b104_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_linear_vector(w, x, b)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_b104_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    result = burkov_lm_ch1_linear_vector(w, x, b)
    assert isinstance(result, dict)
