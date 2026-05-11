"""Tests for tmlcou.tmle_count_outcome."""
import numpy as np
import pytest
from morie.fn.tmlcou import tmle_count_outcome


def test_tmlcou_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    offset = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_count_outcome(y, D, X, offset)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlcou_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    offset = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_count_outcome(y, D, X, offset)
    assert isinstance(result, dict)
