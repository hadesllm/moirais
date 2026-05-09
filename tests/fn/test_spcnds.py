"""Tests for spcnds.schabenberger_conditional_sim."""
import numpy as np
import pytest
from moirais.fn.spcnds import schabenberger_conditional_sim


def test_spcnds_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z_obs = np.random.default_rng(42).normal(0, 1, 100)
    cov_model = 'exponential'
    sim_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_conditional_sim(coords, z_obs, cov_model, sim_grid)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcnds_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    z_obs = np.random.default_rng(42).normal(0, 1, 100)
    cov_model = 'exponential'
    sim_grid = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_conditional_sim(coords, z_obs, cov_model, sim_grid)
    assert isinstance(result, dict)
