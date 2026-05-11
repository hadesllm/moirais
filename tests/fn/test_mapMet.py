"""Tests for mapMet.map_at_k."""
import numpy as np
import pytest
from morie.fn.mapMet import map_at_k


def test_mapMet_basic():
    """Test basic functionality."""
    pred_rank = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = map_at_k(pred_rank, relevant, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mapMet_edge():
    """Test edge cases."""
    pred_rank = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = map_at_k(pred_rank, relevant, k)
    assert isinstance(result, dict)
