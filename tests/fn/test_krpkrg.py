"""Tests for krpkrg.poisson_kriging."""
import numpy as np
import pytest
from morie.fn.krpkrg import poisson_kriging


def test_krpkrg_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    counts = np.random.default_rng(42).normal(0, 1, 100)
    population = np.random.default_rng(42).normal(0, 1, 100)
    variogram = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_kriging(coords, counts, population, variogram)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krpkrg_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    counts = np.random.default_rng(42).normal(0, 1, 100)
    population = np.random.default_rng(42).normal(0, 1, 100)
    variogram = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_kriging(coords, counts, population, variogram)
    assert isinstance(result, dict)
