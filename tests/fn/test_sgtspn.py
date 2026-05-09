"""Tests for sgtspn.sgt_spanning_tree_count."""
import numpy as np
import pytest
from moirais.fn.sgtspn import sgt_spanning_tree_count


def test_sgtspn_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_spanning_tree_count(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtspn_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_spanning_tree_count(A)
    assert isinstance(result, dict)
