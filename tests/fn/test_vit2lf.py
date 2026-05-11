"""Tests for vit2lf.vit2_log_attention."""
import numpy as np
import pytest
from morie.fn.vit2lf import vit2_log_attention


def test_vit2lf_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = vit2_log_attention(q, k, v)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vit2lf_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    result = vit2_log_attention(q, k, v)
    assert isinstance(result, dict)
