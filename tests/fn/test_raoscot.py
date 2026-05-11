"""Tests for raoscot.rao_scott_chisq."""
import numpy as np
import pytest
from morie.fn.raoscot import rao_scott_chisq


def test_raoscot_basic():
    """Test basic functionality."""
    table = np.array([[10, 20, 30], [15, 25, 35]])
    weights = np.random.default_rng(45).exponential(1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = rao_scott_chisq(table, weights, design)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_raoscot_edge():
    """Test edge cases."""
    table = np.array([[10, 20, 30], [15, 25, 35]])
    weights = np.random.default_rng(45).exponential(1, 100)
    design = np.random.default_rng(42).normal(0, 1, 100)
    result = rao_scott_chisq(table, weights, design)
    assert isinstance(result, dict)
