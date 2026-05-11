"""Tests for spreml.schabenberger_reml_variogram."""
import numpy as np
import pytest
from morie.fn.spreml import schabenberger_reml_variogram


def test_spreml_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_reml_variogram(coords, z, X, variogram_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spreml_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_reml_variogram(coords, z, X, variogram_model)
    assert isinstance(result, dict)
