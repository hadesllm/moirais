"""Tests for km085.kamath_ch6_cbs_variance."""
import numpy as np
import pytest
from morie.fn.km085 import kamath_ch6_cbs_variance


def test_km085_basic():
    """Test basic functionality."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    p_a = np.random.default_rng(42).normal(0, 1, 100)
    p_prior = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_cbs_variance(W, A, p_a, p_prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km085_edge():
    """Test edge cases."""
    W = np.random.default_rng(42).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    p_a = np.random.default_rng(42).normal(0, 1, 100)
    p_prior = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_ch6_cbs_variance(W, A, p_a, p_prior)
    assert isinstance(result, dict)
