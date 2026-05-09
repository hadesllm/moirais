"""Tests for deopti.differential_evolution."""
import numpy as np
import pytest
from moirais.fn.deopti import differential_evolution


def test_deopti_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    population = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    CR = np.random.default_rng(42).normal(0, 1, 100)
    result = differential_evolution(f, population, F, CR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_deopti_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    population = np.random.default_rng(42).normal(0, 1, 100)
    F = np.random.default_rng(43).normal(0, 1, 100)
    CR = np.random.default_rng(42).normal(0, 1, 100)
    result = differential_evolution(f, population, F, CR)
    assert isinstance(result, dict)
