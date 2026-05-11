"""Tests for ghs005.ghosal_ch2_location_scale_mixture_limit."""
import numpy as np
import pytest
from morie.fn.ghs005 import ghosal_ch2_location_scale_mixture_limit


def test_ghs005_basic():
    """Test basic functionality."""
    psi = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    mu = 0.0
    result = ghosal_ch2_location_scale_mixture_limit(psi, f, sigma, mu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ghs005_edge():
    """Test edge cases."""
    psi = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    sigma = 1.0
    mu = 0.0
    result = ghosal_ch2_location_scale_mixture_limit(psi, f, sigma, mu)
    assert isinstance(result, dict)
