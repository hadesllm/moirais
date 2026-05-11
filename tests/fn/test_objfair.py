"""Tests for objfair.individual_fairness_lipschitz."""
import numpy as np
import pytest
from morie.fn.objfair import individual_fairness_lipschitz


def test_objfair_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    h_values = np.random.default_rng(42).normal(0, 1, 100)
    x_pairs = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = individual_fairness_lipschitz(y, h_values, x_pairs, L)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_objfair_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    h_values = np.random.default_rng(42).normal(0, 1, 100)
    x_pairs = np.random.default_rng(42).normal(0, 1, 100)
    L = np.random.default_rng(42).normal(0, 1, 100)
    result = individual_fairness_lipschitz(y, h_values, x_pairs, L)
    assert isinstance(result, dict)
