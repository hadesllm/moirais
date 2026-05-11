"""Tests for alrck.alammar_recall_at_k."""
import numpy as np
import pytest
from morie.fn.alrck import alammar_recall_at_k


def test_alrck_basic():
    """Test basic functionality."""
    retrieved = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = alammar_recall_at_k(retrieved, relevant, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alrck_edge():
    """Test edge cases."""
    retrieved = np.random.default_rng(42).normal(0, 1, 100)
    relevant = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    result = alammar_recall_at_k(retrieved, relevant, k)
    assert isinstance(result, dict)
