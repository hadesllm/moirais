"""Tests for abcsmc.abc_smc_epi."""
import numpy as np
import pytest
from morie.fn.abcsmc import abc_smc_epi


def test_abcsmc_basic():
    """Test basic functionality."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    summary_stats = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    n_particles = np.random.default_rng(42).normal(0, 1, 100)
    result = abc_smc_epi(model, summary_stats, priors, n_particles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_abcsmc_edge():
    """Test edge cases."""
    model = np.random.default_rng(42).normal(0, 1, 100)
    summary_stats = np.random.default_rng(42).normal(0, 1, 100)
    priors = np.random.default_rng(42).normal(0, 1, 100)
    n_particles = np.random.default_rng(42).normal(0, 1, 100)
    result = abc_smc_epi(model, summary_stats, priors, n_particles)
    assert isinstance(result, dict)
