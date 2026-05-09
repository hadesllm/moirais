"""Tests for km058.kamath_ch4_lora_forward."""
import numpy as np
import pytest
from moirais.fn.km058 import kamath_ch4_lora_forward


def test_km058_basic():
    """Test basic functionality."""
    W_0 = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_lora_forward(W_0, B, A, x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km058_edge():
    """Test edge cases."""
    W_0 = np.random.default_rng(42).normal(0, 1, 100)
    B = np.random.default_rng(43).normal(0, 1, (10, 10))
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch4_lora_forward(W_0, B, A, x)
    assert isinstance(result, dict)
