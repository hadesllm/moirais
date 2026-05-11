"""Tests for sinkhd.sinkhorn_distance."""
import numpy as np
import pytest
from morie.fn.sinkhd import sinkhorn_distance


def test_sinkhd_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = sinkhorn_distance(a, b, C, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sinkhd_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = sinkhorn_distance(a, b, C, eps)
    assert isinstance(result, dict)
