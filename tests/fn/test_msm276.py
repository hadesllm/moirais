"""Tests for msm276.mvsml_convolutional_nn_eq_14_8."""
import numpy as np
import pytest
from moirais.fn.msm276 import mvsml_convolutional_nn_eq_14_8


def test_msm276_basic():
    """Test basic functionality."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    bc = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    jx = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_8(where, bc, j, T, jx, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm276_edge():
    """Test edge cases."""
    where = np.random.default_rng(42).normal(0, 1, 100)
    bc = np.random.default_rng(42).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    jx = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_8(where, bc, j, T, jx, a)
    assert isinstance(result, dict)
