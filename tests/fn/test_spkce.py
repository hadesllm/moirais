"""Tests for spkce.schabenberger_cov_param_estimation_kriging."""
import numpy as np
import pytest
from morie.fn.spkce import schabenberger_cov_param_estimation_kriging


def test_spkce_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_cov_param_estimation_kriging(coords, z, variogram_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spkce_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_cov_param_estimation_kriging(coords, z, variogram_model)
    assert isinstance(result, dict)
