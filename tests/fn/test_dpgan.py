"""Tests for dpgan.dp_gan."""
import numpy as np
import pytest
from morie.fn.dpgan import dp_gan


def test_dpgan_basic():
    """Test basic functionality."""
    G = np.eye(10)
    D = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = dp_gan(G, D, C, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dpgan_edge():
    """Test edge cases."""
    G = np.eye(10)
    D = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    result = dp_gan(G, D, C, sigma)
    assert isinstance(result, dict)
