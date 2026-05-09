"""Tests for spgaus.schabenberger_gaussian_variogram."""
import numpy as np
import pytest
from moirais.fn.spgaus import schabenberger_gaussian_variogram


def test_spgaus_basic():
    """Test basic functionality."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_gaussian_variogram(h, nugget, sill, range)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spgaus_edge():
    """Test edge cases."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_gaussian_variogram(h, nugget, sill, range)
    assert isinstance(result, dict)
