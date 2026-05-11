"""Tests for msm257.mvsml_general_eq_1_222."""
import numpy as np
import pytest
from morie.fn.msm257 import mvsml_general_eq_1_222


def test_msm257_basic():
    """Test basic functionality."""
    We = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    see = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    best = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_222(We, can, see, that, the, best)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm257_edge():
    """Test edge cases."""
    We = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    see = np.random.default_rng(42).normal(0, 1, 100)
    that = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    best = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_222(We, can, see, that, the, best)
    assert isinstance(result, dict)
