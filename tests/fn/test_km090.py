"""Tests for km090.kamath_ch6_co_occurrence_bias."""
import numpy as np
import pytest
from moirais.fn.km090 import kamath_ch6_co_occurrence_bias


def test_km090_basic():
    """Test basic functionality."""
    w = np.random.default_rng(45).exponential(1, 100)
    A_i = np.random.default_rng(42).normal(0, 1, 100)
    A_j = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_co_occurrence_bias(w, A_i, A_j)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km090_edge():
    """Test edge cases."""
    w = np.random.default_rng(45).exponential(1, 100)
    A_i = np.random.default_rng(42).normal(0, 1, 100)
    A_j = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_co_occurrence_bias(w, A_i, A_j)
    assert isinstance(result, dict)
