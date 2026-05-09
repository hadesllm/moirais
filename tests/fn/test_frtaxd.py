"""Tests for frtaxd.forest_taxon_diversity."""
import numpy as np
import pytest
from moirais.fn.frtaxd import forest_taxon_diversity


def test_frtaxd_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    species = np.random.default_rng(42).normal(0, 1, 100)
    grid = np.random.default_rng(42).normal(0, 1, 100)
    result = forest_taxon_diversity(coords, species, grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_frtaxd_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    species = np.random.default_rng(42).normal(0, 1, 100)
    grid = np.random.default_rng(42).normal(0, 1, 100)
    result = forest_taxon_diversity(coords, species, grid)
    assert isinstance(result, dict)
