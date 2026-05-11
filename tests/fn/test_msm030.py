"""Tests for msm030.mvsml_linear_mixed_models_eq_5_5."""
import numpy as np
import pytest
from morie.fn.msm030 import mvsml_linear_mixed_models_eq_5_5


def test_msm030_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    GID = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    again = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_5(where, y, GID, are, again, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm030_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    GID = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    again = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_5(where, y, GID, are, again, the)
    assert isinstance(result, dict)
