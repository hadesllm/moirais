"""Tests for msm270.mvsml_convolutional_nn_eq_14_7."""
import numpy as np
import pytest
from moirais.fn.msm270 import mvsml_convolutional_nn_eq_14_7


def test_msm270_basic():
    """Test basic functionality."""
    o = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    cio = np.random.default_rng(42).normal(0, 1, 100)
    L2 = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_7(o, t, where, cio, L2, are)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm270_edge():
    """Test edge cases."""
    o = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    cio = np.random.default_rng(42).normal(0, 1, 100)
    L2 = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_7(o, t, where, cio, L2, are)
    assert isinstance(result, dict)
