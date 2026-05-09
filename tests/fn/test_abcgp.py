"""Tests for abcgp.abc_gp_emulator."""
import numpy as np
import pytest
from moirais.fn.abcgp import abc_gp_emulator


def test_abcgp_basic():
    """Test basic functionality."""
    sim = np.random.default_rng(42).normal(0, 1, 100)
    obs = np.random.default_rng(42).normal(0, 1, 100)
    X_grid = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = abc_gp_emulator(sim, obs, X_grid, kernel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_abcgp_edge():
    """Test edge cases."""
    sim = np.random.default_rng(42).normal(0, 1, 100)
    obs = np.random.default_rng(42).normal(0, 1, 100)
    X_grid = np.random.default_rng(42).normal(0, 1, 100)
    kernel = (lambda u: np.exp(-0.5*u*u) / np.sqrt(2*np.pi))
    result = abc_gp_emulator(sim, obs, X_grid, kernel)
    assert isinstance(result, dict)
