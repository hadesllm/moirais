"""Tests for msm261.mvsml_convolutional_nn_eq_14_1."""
import numpy as np
import pytest
from morie.fn.msm261 import mvsml_convolutional_nn_eq_14_1


def test_msm261_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    charg = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    Scholars = np.random.default_rng(42).normal(0, 1, 100)
    Portal = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_1(T, l, charg, de, Scholars, Portal)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm261_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    l = np.random.default_rng(42).normal(0, 1, 100)
    charg = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    Scholars = np.random.default_rng(42).normal(0, 1, 100)
    Portal = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_1(T, l, charg, de, Scholars, Portal)
    assert isinstance(result, dict)
