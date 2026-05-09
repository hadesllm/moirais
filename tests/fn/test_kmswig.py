"""Tests for kmswig.kamath_swiglu_activation."""
import numpy as np
import pytest
from moirais.fn.kmswig import kamath_swiglu_activation


def test_kmswig_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_swiglu_activation(x, W, V, b, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmswig_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_swiglu_activation(x, W, V, b, c)
    assert isinstance(result, dict)
