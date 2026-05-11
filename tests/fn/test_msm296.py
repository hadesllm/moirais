"""Tests for msm296.mvsml_convolutional_nn_eq_14_14."""
import numpy as np
import pytest
from morie.fn.msm296 import mvsml_convolutional_nn_eq_14_14


def test_msm296_basic():
    """Test basic functionality."""
    length = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    general = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    functional = np.random.default_rng(42).normal(0, 1, 100)
    covariate = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(length, t, general, the, functional, covariate)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm296_edge():
    """Test edge cases."""
    length = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    general = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    functional = np.random.default_rng(42).normal(0, 1, 100)
    covariate = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(length, t, general, the, functional, covariate)
    assert isinstance(result, dict)
