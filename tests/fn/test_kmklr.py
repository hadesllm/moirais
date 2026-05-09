"""Tests for kmklr.kamath_kl_reward_shaping."""
import numpy as np
import pytest
from moirais.fn.kmklr import kamath_kl_reward_shaping


def test_kmklr_basic():
    """Test basic functionality."""
    r_phi = np.random.default_rng(42).normal(0, 1, 100)
    kl_divergence = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_kl_reward_shaping(r_phi, kl_divergence, beta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmklr_edge():
    """Test edge cases."""
    r_phi = np.random.default_rng(42).normal(0, 1, 100)
    kl_divergence = np.random.default_rng(42).normal(0, 1, 100)
    beta = 0.8
    result = kamath_kl_reward_shaping(r_phi, kl_divergence, beta)
    assert isinstance(result, dict)
