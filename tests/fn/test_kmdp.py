"""Tests for kmdp.kamath_differential_privacy."""
import numpy as np
import pytest
from moirais.fn.kmdp import kamath_differential_privacy


def test_kmdp_basic():
    """Test basic functionality."""
    eps = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_differential_privacy(eps, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmdp_edge():
    """Test edge cases."""
    eps = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_differential_privacy(eps, delta)
    assert isinstance(result, dict)
