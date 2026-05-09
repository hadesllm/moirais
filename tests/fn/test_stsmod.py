"""Tests for stsmod.state_space_model."""
import numpy as np
import pytest
from moirais.fn.stsmod import state_space_model


def test_stsmod_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    T = np.random.default_rng(43).integers(0, 2, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = state_space_model(y, Z, T, H, Q, R)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stsmod_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Z = np.random.default_rng(43).normal(0, 1, (100, 10))
    T = np.random.default_rng(43).integers(0, 2, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    Q = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    result = state_space_model(y, Z, T, H, Q, R)
    assert isinstance(result, dict)
