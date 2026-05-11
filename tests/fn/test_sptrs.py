"""Tests for sptrs.schabenberger_trend_surface."""
import numpy as np
import pytest
from morie.fn.sptrs import schabenberger_trend_surface


def test_sptrs_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    poly_degree = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_trend_surface(coords, z, poly_degree)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sptrs_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    poly_degree = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_trend_surface(coords, z, poly_degree)
    assert isinstance(result, dict)
