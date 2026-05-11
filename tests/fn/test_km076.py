"""Tests for km076.kamath_ch5_dpo_loss."""
import numpy as np
import pytest
from morie.fn.km076 import kamath_ch5_dpo_loss


def test_km076_basic():
    """Test basic functionality."""
    pi_theta = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_dpo_loss(pi_theta, pi_ref, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_km076_edge():
    """Test edge cases."""
    pi_theta = np.random.default_rng(42).normal(0, 1, 100)
    pi_ref = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_ch5_dpo_loss(pi_theta, pi_ref, beta)
    assert isinstance(result, dict)
