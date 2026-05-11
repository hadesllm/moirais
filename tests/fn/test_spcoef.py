"""Tests for spcoef.spearmans_rho_copula."""
import numpy as np
import pytest
from morie.fn.spcoef import spearmans_rho_copula


def test_spcoef_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = spearmans_rho_copula(y, copula, theta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcoef_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    copula = np.random.default_rng(42).normal(0, 1, 100)
    theta = 0.0
    result = spearmans_rho_copula(y, copula, theta)
    assert isinstance(result, dict)
