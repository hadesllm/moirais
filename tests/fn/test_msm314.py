"""Tests for msm314.mvsml_convolutional_nn_eq_14_14."""
import numpy as np
import pytest
from moirais.fn.msm314 import mvsml_convolutional_nn_eq_14_14


def test_msm314_basic():
    """Test basic functionality."""
    again = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    without = np.random.default_rng(42).normal(0, 1, 100)
    penalization = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(again, the, Bayesian, model, without, penalization)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm314_edge():
    """Test edge cases."""
    again = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    without = np.random.default_rng(42).normal(0, 1, 100)
    penalization = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(again, the, Bayesian, model, without, penalization)
    assert isinstance(result, dict)
