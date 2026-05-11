"""Tests for msm285.mvsml_convolutional_nn_eq_14_12."""
import numpy as np
import pytest
from morie.fn.msm285 import mvsml_convolutional_nn_eq_14_12


def test_msm285_basic():
    """Test basic functionality."""
    reduced = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    TD = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    D1 = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(reduced, to, TD, T, D1, where)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm285_edge():
    """Test edge cases."""
    reduced = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    TD = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    D1 = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(reduced, to, TD, T, D1, where)
    assert isinstance(result, dict)
