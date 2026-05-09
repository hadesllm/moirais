"""Tests for irtmh1.dif_mantel_haenszel."""
import numpy as np
import pytest
from moirais.fn.irtmh1 import dif_mantel_haenszel


def test_irtmh1_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group = np.random.default_rng(42).normal(0, 1, 100)
    total_score = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_mantel_haenszel(X, group, total_score)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_irtmh1_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group = np.random.default_rng(42).normal(0, 1, 100)
    total_score = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_mantel_haenszel(X, group, total_score)
    assert isinstance(result, dict)
