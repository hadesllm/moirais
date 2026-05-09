"""Tests for msm266.mvsml_convolutional_nn_eq_14_2."""
import numpy as np
import pytest
from moirais.fn.msm266 import mvsml_convolutional_nn_eq_14_2


def test_msm266_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    L1 = np.random.default_rng(42).normal(0, 1, 100)
    So = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_2(x, t, l, dt, L1, So)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm266_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    L1 = np.random.default_rng(42).normal(0, 1, 100)
    So = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_2(x, t, l, dt, L1, So)
    assert isinstance(result, dict)
