"""Tests for bndngt.bound_neg_treatment."""
import numpy as np
import pytest
from morie.fn.bndngt import bound_neg_treatment


def test_bndngt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    result = bound_neg_treatment(y, D, y_min)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bndngt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    result = bound_neg_treatment(y, D, y_min)
    assert isinstance(result, dict)
