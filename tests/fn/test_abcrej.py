"""Tests for abcrej.abc_rejection."""
import numpy as np
import pytest
from morie.fn.abcrej import abc_rejection


def test_abcrej_basic():
    """Test basic functionality."""
    sim = np.random.default_rng(42).normal(0, 1, 100)
    obs = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = abc_rejection(sim, obs, eps, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_abcrej_edge():
    """Test edge cases."""
    sim = np.random.default_rng(42).normal(0, 1, 100)
    obs = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = abc_rejection(sim, obs, eps, prior)
    assert isinstance(result, dict)
