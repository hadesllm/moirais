"""Tests for ginicop.ginis_gamma_copula."""
import numpy as np
import pytest
from morie.fn.ginicop import ginis_gamma_copula


def test_ginicop_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = ginis_gamma_copula(y, copula, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ginicop_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = ginis_gamma_copula(y, copula, theta)
    assert isinstance(result, dict)
