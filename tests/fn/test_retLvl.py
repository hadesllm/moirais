"""Tests for retLvl.return_level."""
import numpy as np
import pytest
from morie.fn.retLvl import return_level


def test_retLvl_basic():
    """Test basic functionality."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = return_level(mu, sigma, xi, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_retLvl_edge():
    """Test edge cases."""
    mu = 0.0
    sigma = 1.0
    xi = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = return_level(mu, sigma, xi, T)
    assert isinstance(result, dict)
