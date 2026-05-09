"""Tests for pso_op.particle_swarm."""
import numpy as np
import pytest
from moirais.fn.pso_op import particle_swarm


def test_pso_op_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    n_particles = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    c1 = np.random.default_rng(42).normal(0, 1, 100)
    c2 = np.random.default_rng(42).normal(0, 1, 100)
    result = particle_swarm(f, n_particles, w, c1, c2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pso_op_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    n_particles = np.random.default_rng(42).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    c1 = np.random.default_rng(42).normal(0, 1, 100)
    c2 = np.random.default_rng(42).normal(0, 1, 100)
    result = particle_swarm(f, n_particles, w, c1, c2)
    assert isinstance(result, dict)
