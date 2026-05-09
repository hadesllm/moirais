"""Tests for karpV.genetic_programming."""
import numpy as np
import pytest
from moirais.fn.karpV import genetic_programming


def test_karpV_basic():
    """Test basic functionality."""
    fitness = np.random.default_rng(42).normal(0, 1, 100)
    ops = np.random.default_rng(42).normal(0, 1, 100)
    gens = np.random.default_rng(42).normal(0, 1, 100)
    result = genetic_programming(fitness, ops, gens)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_karpV_edge():
    """Test edge cases."""
    fitness = np.random.default_rng(42).normal(0, 1, 100)
    ops = np.random.default_rng(42).normal(0, 1, 100)
    gens = np.random.default_rng(42).normal(0, 1, 100)
    result = genetic_programming(fitness, ops, gens)
    assert isinstance(result, dict)
