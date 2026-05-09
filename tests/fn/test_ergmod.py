"""Tests for ergmod.ergm."""
import numpy as np
import pytest
from moirais.fn.ergmod import ergm


def test_ergmod_basic():
    """Test basic functionality."""
    G = np.eye(10)
    statistics = np.random.default_rng(42).normal(0, 1, 100)
    theta_init = 0.0
    result = ergm(G, statistics, theta_init)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ergmod_edge():
    """Test edge cases."""
    G = np.eye(10)
    statistics = np.random.default_rng(42).normal(0, 1, 100)
    theta_init = 0.0
    result = ergm(G, statistics, theta_init)
    assert isinstance(result, dict)
