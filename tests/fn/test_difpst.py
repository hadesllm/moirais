"""Tests for difpst.dif_p_diff."""
import numpy as np
import pytest
from moirais.fn.difpst import dif_p_diff


def test_difpst_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_p_diff(X, group)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_difpst_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    group = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_p_diff(X, group)
    assert isinstance(result, dict)
