"""Tests for msm308.mvsml_convolutional_nn_eq_14_14."""
import numpy as np
import pytest
from morie.fn.msm308 import mvsml_convolutional_nn_eq_14_14


def test_msm308_basic():
    """Test basic functionality."""
    ediction = np.random.default_rng(42).normal(0, 1, 100)
    performance = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    no = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(ediction, performance, of, the, model, no)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm308_edge():
    """Test edge cases."""
    ediction = np.random.default_rng(42).normal(0, 1, 100)
    performance = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    no = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(ediction, performance, of, the, model, no)
    assert isinstance(result, dict)
