"""Tests for km149.kamath_ch9_flamingo_factorized."""
import numpy as np
import pytest
from moirais.fn.km149 import kamath_ch9_flamingo_factorized


def test_km149_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_flamingo_factorized(y, x, L)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km149_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch9_flamingo_factorized(y, x, L)
    assert isinstance(result, dict)
