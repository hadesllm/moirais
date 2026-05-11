"""Tests for spglmk.schabenberger_spatial_glm_kriging."""
import numpy as np
import pytest
from morie.fn.spglmk import schabenberger_spatial_glm_kriging


def test_spglmk_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    target = np.random.default_rng(43).integers(0, 2, 100)
    cov_model = 'exponential'
    link = 'identity'
    result = schabenberger_spatial_glm_kriging(x, y, coords, target, cov_model, link)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spglmk_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    target = np.random.default_rng(43).integers(0, 2, 100)
    cov_model = 'exponential'
    link = 'identity'
    result = schabenberger_spatial_glm_kriging(x, y, coords, target, cov_model, link)
    assert isinstance(result, dict)
