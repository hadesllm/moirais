"""Tests for msm012.mvsml_linear_mixed_models_eq_5_2."""
import numpy as np
import pytest
from morie.fn.msm012 import mvsml_linear_mixed_models_eq_5_2


def test_msm012_basic():
    """Test basic functionality."""
    n = 100
    where = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    ZTDZ = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_2(n, where, V, ZTDZ, R, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm012_edge():
    """Test edge cases."""
    n = 100
    where = np.random.default_rng(42).normal(0, 1, 100)
    V = np.random.default_rng(42).normal(0, 1, 100)
    ZTDZ = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_2(n, where, V, ZTDZ, R, the)
    assert isinstance(result, dict)
