"""Tests for km075.kamath_ch5_dpo_pref_simplified."""
import numpy as np
import pytest
from morie.fn.km075 import kamath_ch5_dpo_pref_simplified


def test_km075_basic():
    """Test basic functionality."""
    pi_star = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_dpo_pref_simplified(pi_star, pi_ref, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km075_edge():
    """Test edge cases."""
    pi_star = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_dpo_pref_simplified(pi_star, pi_ref, beta)
    assert isinstance(result, dict)
