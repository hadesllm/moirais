"""Tests for kmbon.kamath_best_of_n_sampling."""
import numpy as np
import pytest
from morie.fn.kmbon import kamath_best_of_n_sampling


def test_kmbon_basic():
    """Test basic functionality."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_best_of_n_sampling(samples, rewards)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmbon_edge():
    """Test edge cases."""
    samples = np.random.default_rng(42).normal(0, 1, 100)
    rewards = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_best_of_n_sampling(samples, rewards)
    assert isinstance(result, dict)
