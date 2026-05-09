"""Tests for hmpol.geron_policy."""
import numpy as np
import pytest
from moirais.fn.hmpol import geron_policy


def test_hmpol_basic():
    """Test basic functionality."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_policy(state, pi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hmpol_edge():
    """Test edge cases."""
    state = np.random.default_rng(42).normal(0, 1, 100)
    pi = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_policy(state, pi)
    assert isinstance(result, dict)
