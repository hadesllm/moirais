"""Tests for spdjkr.schabenberger_disjunctive_kriging."""
import numpy as np
import pytest
from moirais.fn.spdjkr import schabenberger_disjunctive_kriging


def test_spdjkr_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    phi_func = (lambda v: v)
    cov_model = 'exponential'
    result = schabenberger_disjunctive_kriging(coords, z, target, phi_func, cov_model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spdjkr_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z = np.random.default_rng(44).normal(0, 1, 100)
    target = np.random.default_rng(43).integers(0, 2, 100)
    phi_func = (lambda v: v)
    cov_model = 'exponential'
    result = schabenberger_disjunctive_kriging(coords, z, target, phi_func, cov_model)
    assert isinstance(result, dict)
