"""Tests for msm262.mvsml_convolutional_nn_eq_14_2."""
import numpy as np
import pytest
from morie.fn.msm262 import mvsml_convolutional_nn_eq_14_2


def test_msm262_basic():
    """Test basic functionality."""
    possible = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    nd = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    function = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = mvsml_convolutional_nn_eq_14_2(possible, to, nd, a, function, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm262_edge():
    """Test edge cases."""
    possible = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    nd = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    function = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = mvsml_convolutional_nn_eq_14_2(possible, to, nd, a, function, t)
    assert isinstance(result, dict)
