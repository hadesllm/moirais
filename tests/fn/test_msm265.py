"""Tests for msm265.mvsml_convolutional_nn_eq_14_1."""
import numpy as np
import pytest
from morie.fn.msm265 import mvsml_convolutional_nn_eq_14_1


def test_msm265_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_1(x, t, l, dt, E, xT)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm265_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    E = np.random.default_rng(42).normal(0, 1, 100)
    xT = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_1(x, t, l, dt, E, xT)
    assert isinstance(result, dict)
