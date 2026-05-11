"""Tests for rgmuap.rangayyan_muap."""
import numpy as np
import pytest
from morie.fn.rgmuap import rangayyan_muap


def test_rgmuap_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    n_fibers = np.random.default_rng(42).normal(0, 1, 100)
    conduction_vel = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_muap(t, n_fibers, conduction_vel)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgmuap_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    n_fibers = np.random.default_rng(42).normal(0, 1, 100)
    conduction_vel = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_muap(t, n_fibers, conduction_vel)
    assert isinstance(result, dict)
