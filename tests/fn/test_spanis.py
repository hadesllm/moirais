"""Tests for spanis.schabenberger_geometric_anisotropy."""
import numpy as np
import pytest
from morie.fn.spanis import schabenberger_geometric_anisotropy


def test_spanis_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    A_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = schabenberger_geometric_anisotropy(coords, z, A_matrix)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spanis_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    A_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = schabenberger_geometric_anisotropy(coords, z, A_matrix)
    assert isinstance(result, dict)
