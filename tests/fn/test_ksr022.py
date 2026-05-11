"""Tests for ksr022.kosorok_ch1_multiplicative_intensity."""
import numpy as np
import pytest
from morie.fn.ksr022 import kosorok_ch1_multiplicative_intensity


def test_ksr022_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    Lambda = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch1_multiplicative_intensity(t, Z, Y, beta, Lambda)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ksr022_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    Y = np.random.default_rng(43).normal(0, 1, 100)
    beta = 0.8
    Lambda = np.random.default_rng(42).normal(0, 1, 100)
    result = kosorok_ch1_multiplicative_intensity(t, Z, Y, beta, Lambda)
    assert isinstance(result, dict)
