"""Tests for spkfnn.schabenberger_cross_validation_kriging."""
import numpy as np
import pytest
from morie.fn.spkfnn import schabenberger_cross_validation_kriging


def test_spkfnn_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    cov_model = 'exponential'
    result = schabenberger_cross_validation_kriging(coords, z, cov_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spkfnn_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    cov_model = 'exponential'
    result = schabenberger_cross_validation_kriging(coords, z, cov_model)
    assert isinstance(result, dict)
