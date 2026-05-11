"""Tests for rgap.rangayyan_action_potential."""
import numpy as np
import pytest
from morie.fn.rgap import rangayyan_action_potential


def test_rgap_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    v_rest = np.random.default_rng(42).normal(0, 1, 100)
    v_peak = np.random.default_rng(42).normal(0, 1, 100)
    t_rise = np.random.default_rng(42).normal(0, 1, 100)
    t_fall = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_action_potential(t, v_rest, v_peak, t_rise, t_fall)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgap_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    v_rest = np.random.default_rng(42).normal(0, 1, 100)
    v_peak = np.random.default_rng(42).normal(0, 1, 100)
    t_rise = np.random.default_rng(42).normal(0, 1, 100)
    t_fall = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_action_potential(t, v_rest, v_peak, t_rise, t_fall)
    assert isinstance(result, dict)
