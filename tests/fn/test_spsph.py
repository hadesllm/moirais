"""Tests for spsph.schabenberger_spherical_variogram."""
import numpy as np
import pytest
from moirais.fn.spsph import schabenberger_spherical_variogram


def test_spsph_basic():
    """Test basic functionality."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_spherical_variogram(h, nugget, sill, range)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spsph_edge():
    """Test edge cases."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_spherical_variogram(h, nugget, sill, range)
    assert isinstance(result, dict)
