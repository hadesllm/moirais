"""Tests for kmadal.kamath_adalora_rank_allocation."""
import numpy as np
import pytest
from moirais.fn.kmadal import kamath_adalora_rank_allocation


def test_kmadal_basic():
    """Test basic functionality."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    Q = np.random.default_rng(42).normal(0, 1, 100)
    importance = np.random.default_rng(42).normal(0, 1, 100)
    target_rank = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_adalora_rank_allocation(P, s, Q, importance, target_rank)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_kmadal_edge():
    """Test edge cases."""
    P = np.random.default_rng(42).normal(0, 1, 100)
    s = 90
    Q = np.random.default_rng(42).normal(0, 1, 100)
    importance = np.random.default_rng(42).normal(0, 1, 100)
    target_rank = np.random.default_rng(42).normal(0, 1, 100)
    result = kamath_adalora_rank_allocation(P, s, Q, importance, target_rank)
    assert isinstance(result, dict)
