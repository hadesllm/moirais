"""Tests for kldivg.kl_divergence."""
import numpy as np
import pytest
from moirais.fn.kldivg import kldivg as kl_divergence


def test_kldivg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = kl_divergence(y, p, q, base)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kldivg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    base = np.random.default_rng(42).normal(0, 1, 100)
    result = kl_divergence(y, p, q, base)
    assert isinstance(result, dict)
