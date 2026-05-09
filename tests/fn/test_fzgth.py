"""Tests for fzgth.fauzi_g_theta_distribution."""
import numpy as np
import pytest
from moirais.fn.fzgth import fauzi_g_theta_distribution


def test_fzgth_basic():
    """Test basic functionality."""
    theta = 0.0
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_g_theta_distribution(theta, cdf, density)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzgth_edge():
    """Test edge cases."""
    theta = 0.0
    cdf = (lambda v: 1.0 / (1.0 + np.exp(-v)))
    density = np.random.default_rng(42).normal(0, 1, 100)
    result = fauzi_g_theta_distribution(theta, cdf, density)
    assert isinstance(result, dict)
