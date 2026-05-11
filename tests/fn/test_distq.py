"""Tests for distq.distributional_rl."""
import numpy as np
import pytest
from morie.fn.distq import distributional_rl


def test_distq_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    atoms = np.random.default_rng(42).normal(0, 1, 100)
    vmin = np.random.default_rng(42).normal(0, 1, 100)
    vmax = np.random.default_rng(42).normal(0, 1, 100)
    result = distributional_rl(env, atoms, vmin, vmax)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_distq_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    atoms = np.random.default_rng(42).normal(0, 1, 100)
    vmin = np.random.default_rng(42).normal(0, 1, 100)
    vmax = np.random.default_rng(42).normal(0, 1, 100)
    result = distributional_rl(env, atoms, vmin, vmax)
    assert isinstance(result, dict)
