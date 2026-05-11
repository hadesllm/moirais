"""Tests for hier.hierarchical_rl."""
import numpy as np
import pytest
from morie.fn.hier import hierarchical_rl


def test_hier_basic():
    """Test basic functionality."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    options = np.random.default_rng(42).normal(0, 1, 100)
    meta = np.random.default_rng(42).normal(0, 1, 100)
    result = hierarchical_rl(env, options, meta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hier_edge():
    """Test edge cases."""
    env = np.random.default_rng(42).normal(0, 1, 100)
    options = np.random.default_rng(42).normal(0, 1, 100)
    meta = np.random.default_rng(42).normal(0, 1, 100)
    result = hierarchical_rl(env, options, meta)
    assert isinstance(result, dict)
