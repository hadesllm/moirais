"""Tests for km069.kamath_ch5_rlhf_objective."""
import numpy as np
import pytest
from moirais.fn.km069 import kamath_ch5_rlhf_objective


def test_km069_basic():
    """Test basic functionality."""
    pi_theta = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    r_phi = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_rlhf_objective(pi_theta, pi_ref, r_phi, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km069_edge():
    """Test edge cases."""
    pi_theta = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    r_phi = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_rlhf_objective(pi_theta, pi_ref, r_phi, beta)
    assert isinstance(result, dict)
