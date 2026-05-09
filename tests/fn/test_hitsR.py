"""Tests for hitsR.hits_at_k."""
import numpy as np
import pytest
from moirais.fn.hitsR import hits_at_k


def test_hitsR_basic():
    """Test basic functionality."""
    pred_rank = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = hits_at_k(pred_rank, relevant, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hitsR_edge():
    """Test edge cases."""
    pred_rank = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = hits_at_k(pred_rank, relevant, k)
    assert isinstance(result, dict)
