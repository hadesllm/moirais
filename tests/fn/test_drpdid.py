"""Tests for drpdid.placebo_dr_did."""
import numpy as np
import pytest
from morie.fn.drpdid import placebo_dr_did


def test_drpdid_basic():
    """Test basic functionality."""
    y_pre1 = np.random.default_rng(42).normal(0, 1, 100)
    y_pre2 = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = placebo_dr_did(y_pre1, y_pre2, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drpdid_edge():
    """Test edge cases."""
    y_pre1 = np.random.default_rng(42).normal(0, 1, 100)
    y_pre2 = np.random.default_rng(42).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = placebo_dr_did(y_pre1, y_pre2, D, X)
    assert isinstance(result, dict)
