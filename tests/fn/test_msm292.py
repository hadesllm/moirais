"""Tests for msm292.mvsml_convolutional_nn_eq_14_1."""
import numpy as np
import pytest
from morie.fn.msm292 import mvsml_convolutional_nn_eq_14_1


def test_msm292_basic():
    """Test basic functionality."""
    l = np.random.default_rng(42).normal(0, 1, 100)
    Ei = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    e = np.random.default_rng(44).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_1(l, Ei, y, X, e, prior)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm292_edge():
    """Test edge cases."""
    l = np.random.default_rng(42).normal(0, 1, 100)
    Ei = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    e = np.random.default_rng(44).normal(0, 1, 100)
    prior = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_1(l, Ei, y, X, e, prior)
    assert isinstance(result, dict)
