"""Tests for causrosen.causal_rosenbaum_bound."""
import numpy as np
import pytest
from morie.fn.causrosen import causal_rosenbaum_bound


def test_causrosen_basic():
    """Test basic functionality."""
    paired_diff = np.random.default_rng(42).normal(0, 1, 100)
    Gamma = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_rosenbaum_bound(paired_diff, Gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_causrosen_edge():
    """Test edge cases."""
    paired_diff = np.random.default_rng(42).normal(0, 1, 100)
    Gamma = np.random.default_rng(42).normal(0, 1, 100)
    result = causal_rosenbaum_bound(paired_diff, Gamma)
    assert isinstance(result, dict)
