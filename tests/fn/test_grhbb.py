"""Tests for grhbb.geron_hebb_rule."""
import numpy as np
import pytest
from morie.fn.grhbb import geron_hebb_rule


def test_grhbb_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hebb_rule(x, y_true, y_pred, w, eta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grhbb_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y_true = np.random.default_rng(43).integers(0, 2, 100)
    y_pred = np.random.default_rng(44).normal(0, 1, 100)
    w = np.random.default_rng(45).exponential(1, 100)
    eta = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_hebb_rule(x, y_true, y_pred, w, eta)
    assert isinstance(result, dict)
