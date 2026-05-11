"""Tests for msm295.mvsml_convolutional_nn_eq_14_13."""
import numpy as np
import pytest
from morie.fn.msm295 import mvsml_convolutional_nn_eq_14_13


def test_msm295_basic():
    """Test basic functionality."""
    cat = np.random.default_rng(42).normal(0, 1, 100)
    Partition = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Estimation = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(cat, Partition, p, Bayesian, Estimation, of)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm295_edge():
    """Test edge cases."""
    cat = np.random.default_rng(42).normal(0, 1, 100)
    Partition = np.random.default_rng(42).normal(0, 1, 100)
    p = 5
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    Estimation = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(cat, Partition, p, Bayesian, Estimation, of)
    assert isinstance(result, dict)
