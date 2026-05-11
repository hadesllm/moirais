"""Tests for fanmd.fanova_decomposition."""
import numpy as np
import pytest
from morie.fn.fanmd import fanova_decomposition


def test_fanmd_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    input_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = fanova_decomposition(f, input_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fanmd_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    input_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = fanova_decomposition(f, input_dist)
    assert isinstance(result, dict)
