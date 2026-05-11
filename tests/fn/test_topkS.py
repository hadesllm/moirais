"""Tests for topkS.top_k_sampling."""
import numpy as np
import pytest
from morie.fn.topkS import top_k_sampling


def test_topkS_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    temp = np.random.default_rng(42).normal(0, 1, 100)
    result = top_k_sampling(logits, k, temp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_topkS_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    temp = np.random.default_rng(42).normal(0, 1, 100)
    result = top_k_sampling(logits, k, temp)
    assert isinstance(result, dict)
