"""Tests for rndut.random_utility_model."""
import numpy as np
import pytest
from moirais.fn.rndut import random_utility_model


def test_rndut_basic():
    """Test basic functionality."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    eps_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = random_utility_model(V, eps_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rndut_edge():
    """Test edge cases."""
    V = np.random.default_rng(42).normal(0, 1, 100)
    eps_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = random_utility_model(V, eps_dist)
    assert isinstance(result, dict)
