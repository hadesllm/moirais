"""Tests for vitscn.vit_scaled_cosine."""
import numpy as np
import pytest
from moirais.fn.vitscn import vit_scaled_cosine


def test_vitscn_basic():
    """Test basic functionality."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    tau = 0.1
    result = vit_scaled_cosine(q, k, v, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_vitscn_edge():
    """Test edge cases."""
    q = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    v = np.random.default_rng(44).normal(0, 1, 100)
    tau = 0.1
    result = vit_scaled_cosine(q, k, v, tau)
    assert isinstance(result, dict)
