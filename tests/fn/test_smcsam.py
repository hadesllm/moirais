"""Tests for smcsam.sequential_mc_sampler."""
import numpy as np
import pytest
from morie.fn.smcsam import sequential_mc_sampler


def test_smcsam_basic():
    """Test basic functionality."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    temperatures = np.random.default_rng(42).normal(0, 1, 100)
    n_particles = np.random.default_rng(42).normal(0, 1, 100)
    result = sequential_mc_sampler(log_p, temperatures, n_particles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smcsam_edge():
    """Test edge cases."""
    log_p = np.random.default_rng(42).normal(0, 1, 100)
    temperatures = np.random.default_rng(42).normal(0, 1, 100)
    n_particles = np.random.default_rng(42).normal(0, 1, 100)
    result = sequential_mc_sampler(log_p, temperatures, n_particles)
    assert isinstance(result, dict)
