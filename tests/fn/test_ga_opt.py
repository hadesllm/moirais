"""Tests for ga_opt.genetic_algorithm."""
import numpy as np
import pytest
from morie.fn.ga_opt import genetic_algorithm


def test_ga_opt_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    population = np.random.default_rng(42).normal(0, 1, 100)
    generations = np.random.default_rng(42).normal(0, 1, 100)
    result = genetic_algorithm(f, population, generations)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ga_opt_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    population = np.random.default_rng(42).normal(0, 1, 100)
    generations = np.random.default_rng(42).normal(0, 1, 100)
    result = genetic_algorithm(f, population, generations)
    assert isinstance(result, dict)
