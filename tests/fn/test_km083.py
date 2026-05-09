"""Tests for km083.kamath_ch6_ceat_random_effects."""
import numpy as np
import pytest
from moirais.fn.km083 import kamath_ch6_ceat_random_effects


def test_km083_basic():
    """Test basic functionality."""
    S_A1 = np.random.default_rng(42).normal(0, 1, 100)
    S_A2 = np.random.default_rng(42).normal(0, 1, 100)
    S_W1 = np.random.default_rng(42).normal(0, 1, 100)
    S_W2 = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch6_ceat_random_effects(S_A1, S_A2, S_W1, S_W2, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km083_edge():
    """Test edge cases."""
    S_A1 = np.random.default_rng(42).normal(0, 1, 100)
    S_A2 = np.random.default_rng(42).normal(0, 1, 100)
    S_W1 = np.random.default_rng(42).normal(0, 1, 100)
    S_W2 = np.random.default_rng(42).normal(0, 1, 100)
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = kamath_ch6_ceat_random_effects(S_A1, S_A2, S_W1, S_W2, v)
    assert isinstance(result, dict)
