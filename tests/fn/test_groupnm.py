"""Tests for groupnm.group_norm."""
import numpy as np
import pytest
from moirais.fn.groupnm import group_norm


def test_groupnm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = group_norm(y, x, groups, g, b, eps)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_groupnm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    groups = np.random.default_rng(43).integers(0, 3, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    eps = np.random.default_rng(42).normal(0, 1, 100)
    result = group_norm(y, x, groups, g, b, eps)
    assert isinstance(result, dict)
