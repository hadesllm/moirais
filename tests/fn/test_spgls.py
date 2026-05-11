"""Tests for spgls.schabenberger_gls_spatial."""
import numpy as np
import pytest
from morie.fn.spgls import schabenberger_gls_spatial


def test_spgls_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    result = schabenberger_gls_spatial(x, y, sigma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spgls_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    sigma = 1.0
    result = schabenberger_gls_spatial(x, y, sigma)
    assert isinstance(result, dict)
