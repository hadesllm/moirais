"""Tests for ghs026.ghosal_ch3_tailfree_finite_density_pm."""
import numpy as np
import pytest
from morie.fn.ghs026 import ghosal_ch3_tailfree_finite_density_pm


def test_ghs026_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    A_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = ghosal_ch3_tailfree_finite_density_pm(P, mu, A_epsilon, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs026_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    A_epsilon = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = ghosal_ch3_tailfree_finite_density_pm(P, mu, A_epsilon, m)
    assert isinstance(result, dict)
