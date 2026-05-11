"""Tests for msm264.mvsml_convolutional_nn_eq_14_3."""
import numpy as np
import pytest
from morie.fn.msm264 import mvsml_convolutional_nn_eq_14_3


def test_msm264_basic():
    """Test basic functionality."""
    Functional = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    function = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    represented = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_3(Functional, Regression, function, to, be, represented)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm264_edge():
    """Test edge cases."""
    Functional = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    function = np.random.default_rng(42).normal(0, 1, 100)
    to = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    represented = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_3(Functional, Regression, function, to, be, represented)
    assert isinstance(result, dict)
