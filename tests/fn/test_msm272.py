"""Tests for msm272.mvsml_convolutional_nn_eq_14_9."""
import numpy as np
import pytest
from moirais.fn.msm272 import mvsml_convolutional_nn_eq_14_9


def test_msm272_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    L1 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    m = 10
    dt = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_9(R, T, L1, t, m, dt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm272_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    L1 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    m = 10
    dt = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_9(R, T, L1, t, m, dt)
    assert isinstance(result, dict)
