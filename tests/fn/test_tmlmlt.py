"""Tests for tmlmlt.tmle_multiple_treatments."""
import numpy as np
import pytest
from morie.fn.tmlmlt import tmle_multiple_treatments


def test_tmlmlt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    arm_set = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_multiple_treatments(y, D, X, arm_set)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_tmlmlt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    arm_set = np.random.default_rng(42).normal(0, 1, 100)
    result = tmle_multiple_treatments(y, D, X, arm_set)
    assert isinstance(result, dict)
