"""Tests for kmtot.kamath_tree_of_thoughts."""
import numpy as np
import pytest
from morie.fn.kmtot import kamath_tree_of_thoughts


def test_kmtot_basic():
    """Test basic functionality."""
    problem = np.random.default_rng(42).normal(0, 1, 100)
    branch_factor = np.random.default_rng(42).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_tree_of_thoughts(problem, branch_factor, max_depth, model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmtot_edge():
    """Test edge cases."""
    problem = np.random.default_rng(42).normal(0, 1, 100)
    branch_factor = np.random.default_rng(42).normal(0, 1, 100)
    max_depth = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_tree_of_thoughts(problem, branch_factor, max_depth, model)
    assert isinstance(result, dict)
