"""Tests for rgbhatt.rangayyan_bhattacharyya."""
import numpy as np
import pytest
from morie.fn.rgbhatt import rangayyan_bhattacharyya


def test_rgbhatt_basic():
    """Test basic functionality."""
    mu1 = np.random.default_rng(42).normal(0, 1, 100)
    sigma1 = np.random.default_rng(42).normal(0, 1, 100)
    mu2 = np.random.default_rng(42).normal(0, 1, 100)
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bhattacharyya(mu1, sigma1, mu2, sigma2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgbhatt_edge():
    """Test edge cases."""
    mu1 = np.random.default_rng(42).normal(0, 1, 100)
    sigma1 = np.random.default_rng(42).normal(0, 1, 100)
    mu2 = np.random.default_rng(42).normal(0, 1, 100)
    sigma2 = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_bhattacharyya(mu1, sigma1, mu2, sigma2)
    assert isinstance(result, dict)
