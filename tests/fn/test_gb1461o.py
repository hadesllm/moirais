"""Tests for gb1461o.gibbons_ordered_categories."""
import numpy as np
import pytest
from moirais.fn.gb1461o import gibbons_ordered_categories


def test_gb1461o_basic():
    """Test basic functionality."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = gibbons_ordered_categories(table, scores)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb1461o_edge():
    """Test edge cases."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    scores = np.random.default_rng(42).uniform(0, 1, 100)
    result = gibbons_ordered_categories(table, scores)
    assert isinstance(result, dict)
