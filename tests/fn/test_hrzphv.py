"""Tests for hrzphv.horowitz_ph_heterogeneity."""
import numpy as np
import pytest
from morie.fn.hrzphv import horowitz_ph_heterogeneity


def test_hrzphv_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    frailty_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_ph_heterogeneity(t, x, event, frailty_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hrzphv_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    event = np.random.default_rng(42).normal(0, 1, 100)
    frailty_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = horowitz_ph_heterogeneity(t, x, event, frailty_dist)
    assert isinstance(result, dict)
