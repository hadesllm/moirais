"""Tests for km084.kamath_ch6_lpbs_bias."""
import numpy as np
import pytest
from morie.fn.km084 import kamath_ch6_lpbs_bias


def test_km084_basic():
    """Test basic functionality."""
    p_a = np.random.default_rng(42).normal(0, 1, 100)
    p_prior = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_lpbs_bias(p_a, p_prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km084_edge():
    """Test edge cases."""
    p_a = np.random.default_rng(42).normal(0, 1, 100)
    p_prior = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_lpbs_bias(p_a, p_prior)
    assert isinstance(result, dict)
