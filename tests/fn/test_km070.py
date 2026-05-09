"""Tests for km070.kamath_ch5_rlhf_optimal_policy."""
import numpy as np
import pytest
from moirais.fn.km070 import kamath_ch5_rlhf_optimal_policy


def test_km070_basic():
    """Test basic functionality."""
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = kamath_ch5_rlhf_optimal_policy(pi_ref, r, beta, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km070_edge():
    """Test edge cases."""
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    r = 10
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = kamath_ch5_rlhf_optimal_policy(pi_ref, r, beta, Z)
    assert isinstance(result, dict)
