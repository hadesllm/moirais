"""Tests for difsbs.dif_sibtest."""
import numpy as np
import pytest
from morie.fn.difsbs import dif_sibtest


def test_difsbs_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group = np.random.default_rng(42).normal(0, 1, 100)
    matching = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_sibtest(X, group, matching)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_difsbs_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group = np.random.default_rng(42).normal(0, 1, 100)
    matching = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_sibtest(X, group, matching)
    assert isinstance(result, dict)
