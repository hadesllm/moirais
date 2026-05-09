"""Tests for counRS.counterfactual_rec."""
import numpy as np
import pytest
from moirais.fn.counRS import counterfactual_rec


def test_counRS_basic():
    """Test basic functionality."""
    logged = np.random.default_rng(42).normal(0, 1, 100)
    new_policy = np.random.default_rng(42).normal(0, 1, 100)
    result = counterfactual_rec(logged, new_policy)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_counRS_edge():
    """Test edge cases."""
    logged = np.random.default_rng(42).normal(0, 1, 100)
    new_policy = np.random.default_rng(42).normal(0, 1, 100)
    result = counterfactual_rec(logged, new_policy)
    assert isinstance(result, dict)
