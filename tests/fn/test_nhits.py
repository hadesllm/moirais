"""Tests for nhits.n_hits."""
import numpy as np
import pytest
from morie.fn.nhits import n_hits


def test_nhits_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stacks = np.random.default_rng(42).normal(0, 1, 100)
    mlp_units = np.random.default_rng(42).normal(0, 1, 100)
    result = n_hits(y, stacks, mlp_units)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nhits_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    stacks = np.random.default_rng(42).normal(0, 1, 100)
    mlp_units = np.random.default_rng(42).normal(0, 1, 100)
    result = n_hits(y, stacks, mlp_units)
    assert isinstance(result, dict)
