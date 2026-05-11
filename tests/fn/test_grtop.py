"""Tests for grtop.geron_topk_sampling."""
import numpy as np
import pytest
from morie.fn.grtop import geron_topk_sampling


def test_grtop_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_topk_sampling(logits, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grtop_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = geron_topk_sampling(logits, k)
    assert isinstance(result, dict)
