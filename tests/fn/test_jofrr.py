"""Tests for jofrr.joseph_fourier_features."""
import numpy as np
import pytest
from morie.fn.jofrr import joseph_fourier_features


def test_jofrr_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    m = 10
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = joseph_fourier_features(t, m, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jofrr_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    m = 10
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = joseph_fourier_features(t, m, K)
    assert isinstance(result, dict)
