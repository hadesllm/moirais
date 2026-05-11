"""Tests for spkpe.schabenberger_kriging_pred_error."""
import numpy as np
import pytest
from morie.fn.spkpe import schabenberger_kriging_pred_error


def test_spkpe_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_kriging_pred_error(coords, z, target, variogram_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spkpe_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    variogram_model = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_kriging_pred_error(coords, z, target, variogram_model)
    assert isinstance(result, dict)
