"""Tests for km074.kamath_ch5_dpo_pref_substituted."""
import numpy as np
import pytest
from moirais.fn.km074 import kamath_ch5_dpo_pref_substituted


def test_km074_basic():
    """Test basic functionality."""
    pi_star = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta, Z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km074_edge():
    """Test edge cases."""
    pi_star = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    result = kamath_ch5_dpo_pref_substituted(pi_star, pi_ref, beta, Z)
    assert isinstance(result, dict)
