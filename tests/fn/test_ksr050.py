"""Tests for ksr050.kosorok_ch2_frechet_differentiability."""
import numpy as np
import pytest
from moirais.fn.ksr050 import kosorok_ch2_frechet_differentiability


def test_ksr050_basic():
    """Test basic functionality."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    h_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_frechet_differentiability(phi, theta, h_n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr050_edge():
    """Test edge cases."""
    phi = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    h_n = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch2_frechet_differentiability(phi, theta, h_n)
    assert isinstance(result, dict)
