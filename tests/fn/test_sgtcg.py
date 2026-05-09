"""Tests for sgtcg.sgt_commute_distance."""
import numpy as np
import pytest
from moirais.fn.sgtcg import sgt_commute_distance


def test_sgtcg_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_commute_distance(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtcg_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_commute_distance(A)
    assert isinstance(result, dict)
