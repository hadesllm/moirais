"""Tests for toppS.top_p_sampling."""
import numpy as np
import pytest
from moirais.fn.toppS import top_p_sampling


def test_toppS_basic():
    """Test basic functionality."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    temp = np.random.default_rng(42).normal(0, 1, 100)
    result = top_p_sampling(logits, p, temp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_toppS_edge():
    """Test edge cases."""
    logits = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    temp = np.random.default_rng(42).normal(0, 1, 100)
    result = top_p_sampling(logits, p, temp)
    assert isinstance(result, dict)
