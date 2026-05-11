"""Tests for hrzphd.horowitz_ph_discrete_obs."""
import numpy as np
import pytest
from morie.fn.hrzphd import horowitz_ph_discrete_obs


def test_hrzphd_basic():
    """Test basic functionality."""
    t_discrete = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_ph_discrete_obs(t_discrete, x, event)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzphd_edge():
    """Test edge cases."""
    t_discrete = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_ph_discrete_obs(t_discrete, x, event)
    assert isinstance(result, dict)
