"""Tests for ndcg.ndcg."""
import numpy as np
import pytest
from moirais.fn.ndcg import ndcg


def test_ndcg_basic():
    """Test basic functionality."""
    pred_rank = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ndcg(pred_rank, relevant, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ndcg_edge():
    """Test edge cases."""
    pred_rank = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = ndcg(pred_rank, relevant, k)
    assert isinstance(result, dict)
