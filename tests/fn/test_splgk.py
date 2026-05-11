"""Tests for splgk.schabenberger_lognormal_kriging."""
import numpy as np
import pytest
from morie.fn.splgk import schabenberger_lognormal_kriging


def test_splgk_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    cov_model = 'exponential'
    result = schabenberger_lognormal_kriging(coords, z, target, cov_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_splgk_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    cov_model = 'exponential'
    result = schabenberger_lognormal_kriging(coords, z, target, cov_model)
    assert isinstance(result, dict)
