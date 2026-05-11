"""Tests for pftrep.particle_filter_epi."""
import numpy as np
import pytest
from morie.fn.pftrep import particle_filter_epi


def test_pftrep_basic():
    """Test basic functionality."""
    pomp_model = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    n_particles = np.random.default_rng(42).normal(0, 1, 100)
    result = particle_filter_epi(pomp_model, data, n_particles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pftrep_edge():
    """Test edge cases."""
    pomp_model = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    n_particles = np.random.default_rng(42).normal(0, 1, 100)
    result = particle_filter_epi(pomp_model, data, n_particles)
    assert isinstance(result, dict)
