"""Tests for spexp.schabenberger_exponential_variogram."""
import numpy as np
import pytest
from morie.fn.spexp import schabenberger_exponential_variogram


def test_spexp_basic():
    """Test basic functionality."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_exponential_variogram(h, nugget, sill, range)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spexp_edge():
    """Test edge cases."""
    h = 0.3
    nugget = np.random.default_rng(42).normal(0, 1, 100)
    sill = np.random.default_rng(42).normal(0, 1, 100)
    range = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_exponential_variogram(h, nugget, sill, range)
    assert isinstance(result, dict)
