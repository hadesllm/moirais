"""Tests for vitatt.vit_self_attention."""
import numpy as np
import pytest
from moirais.fn.vitatt import vit_self_attention


def test_vitatt_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = vit_self_attention(q, k, v, mask)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vitatt_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    mask = np.random.default_rng(42).normal(0, 1, 100)
    result = vit_self_attention(q, k, v, mask)
    assert isinstance(result, dict)
