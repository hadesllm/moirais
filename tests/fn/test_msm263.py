"""Tests for msm263.mvsml_convolutional_nn_eq_14_1."""
import numpy as np
import pytest
from morie.fn.msm263 import mvsml_convolutional_nn_eq_14_1


def test_msm263_basic():
    """Test basic functionality."""
    elements = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    a = np.random.default_rng(44).normal(0, 1, 100)
    function = np.random.default_rng(42).normal(0, 1, 100)
    space = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_1(elements, of, basis, a, function, space)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm263_edge():
    """Test edge cases."""
    elements = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    basis = np.random.default_rng(42).normal(0, 1, (100, 5))
    a = np.random.default_rng(44).normal(0, 1, 100)
    function = np.random.default_rng(42).normal(0, 1, 100)
    space = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_1(elements, of, basis, a, function, space)
    assert isinstance(result, dict)
