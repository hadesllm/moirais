"""Tests for msm282.mvsml_convolutional_nn_eq_14_10."""
import numpy as np
import pytest
from morie.fn.msm282 import mvsml_convolutional_nn_eq_14_10


def test_msm282_basic():
    """Test basic functionality."""
    p = 5
    t = np.linspace(0, 10, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    derivate = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = mvsml_convolutional_nn_eq_14_10(p, t, a, derivate, of, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm282_edge():
    """Test edge cases."""
    p = 5
    t = np.linspace(0, 10, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    derivate = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = mvsml_convolutional_nn_eq_14_10(p, t, a, derivate, of, order)
    assert isinstance(result, dict)
