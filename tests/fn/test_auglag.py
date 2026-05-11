"""Tests for auglag.augmented_lagrangian."""
import numpy as np
import pytest
from morie.fn.auglag import augmented_lagrangian


def test_auglag_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    lambda0 = np.random.default_rng(42).normal(0, 1, 100)
    result = augmented_lagrangian(f, constraints, x0, mu, lambda0)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_auglag_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    constraints = np.random.default_rng(42).normal(0, 1, 100)
    x0 = np.random.default_rng(42).normal(0, 1, 100)
    mu = 0.0
    lambda0 = np.random.default_rng(42).normal(0, 1, 100)
    result = augmented_lagrangian(f, constraints, x0, mu, lambda0)
    assert isinstance(result, dict)
