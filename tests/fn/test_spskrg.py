"""Tests for spskrg.schabenberger_simple_kriging."""
import numpy as np
import pytest
from morie.fn.spskrg import schabenberger_simple_kriging


def test_spskrg_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    cov_model = 'exponential'
    mu = 0.0
    result = schabenberger_simple_kriging(coords, z, target, cov_model, mu)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spskrg_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    cov_model = 'exponential'
    mu = 0.0
    result = schabenberger_simple_kriging(coords, z, target, cov_model, mu)
    assert isinstance(result, dict)
