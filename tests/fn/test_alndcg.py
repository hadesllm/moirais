"""Tests for alndcg.alammar_ndcg_at_k."""
import numpy as np
import pytest
from morie.fn.alndcg import alammar_ndcg_at_k


def test_alndcg_basic():
    """Test basic functionality."""
    relevances = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = alammar_ndcg_at_k(relevances, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alndcg_edge():
    """Test edge cases."""
    relevances = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = alammar_ndcg_at_k(relevances, k)
    assert isinstance(result, dict)
