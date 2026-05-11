"""Tests for drspr.dr_spillover."""
import numpy as np
import pytest
from morie.fn.drspr import dr_spillover


def test_drspr_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    exposure = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_spillover(y, D, X, exposure)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drspr_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    exposure = np.random.default_rng(42).normal(0, 1, 100)
    result = dr_spillover(y, D, X, exposure)
    assert isinstance(result, dict)
