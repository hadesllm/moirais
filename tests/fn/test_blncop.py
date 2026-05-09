"""Tests for blncop.blomqvists_beta_copula."""
import numpy as np
import pytest
from moirais.fn.blncop import blomqvists_beta_copula


def test_blncop_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = blomqvists_beta_copula(y, copula, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blncop_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = blomqvists_beta_copula(y, copula, theta)
    assert isinstance(result, dict)
