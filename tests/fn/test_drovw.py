"""Tests for drovw.dr_overlap_weighted."""
import numpy as np
import pytest
from moirais.fn.drovw import dr_overlap_weighted


def test_drovw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_overlap_weighted(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drovw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_overlap_weighted(y, D, X)
    assert isinstance(result, dict)
