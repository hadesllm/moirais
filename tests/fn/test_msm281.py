"""Tests for msm281.mvsml_convolutional_nn_eq_14_11."""
import numpy as np
import pytest
from moirais.fn.msm281 import mvsml_convolutional_nn_eq_14_11


def test_msm281_basic():
    """Test basic functionality."""
    p = 5
    t = np.linspace(0, 10, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    derivate = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = mvsml_convolutional_nn_eq_14_11(p, t, a, derivate, of, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm281_edge():
    """Test edge cases."""
    p = 5
    t = np.linspace(0, 10, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    derivate = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = mvsml_convolutional_nn_eq_14_11(p, t, a, derivate, of, order)
    assert isinstance(result, dict)
