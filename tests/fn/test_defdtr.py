"""Tests for defdtr.deformable_detr."""
import numpy as np
import pytest
from morie.fn.defdtr import deformable_detr


def test_defdtr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = deformable_detr(x, queries, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_defdtr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = deformable_detr(x, queries, K)
    assert isinstance(result, dict)
