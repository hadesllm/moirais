"""Tests for msm293.mvsml_convolutional_nn_eq_14_13."""
import numpy as np
import pytest
from morie.fn.msm293 import mvsml_convolutional_nn_eq_14_13


def test_msm293_basic():
    """Test basic functionality."""
    random = np.random.default_rng(42).normal(0, 1, 100)
    partitions = np.random.default_rng(42).normal(0, 1, 100)
    they = np.random.default_rng(42).normal(0, 1, 100)
    were = np.random.default_rng(42).normal(0, 1, 100)
    better = np.random.default_rng(42).normal(0, 1, 100)
    than = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(random, partitions, they, were, better, than)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm293_edge():
    """Test edge cases."""
    random = np.random.default_rng(42).normal(0, 1, 100)
    partitions = np.random.default_rng(42).normal(0, 1, 100)
    they = np.random.default_rng(42).normal(0, 1, 100)
    were = np.random.default_rng(42).normal(0, 1, 100)
    better = np.random.default_rng(42).normal(0, 1, 100)
    than = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(random, partitions, they, were, better, than)
    assert isinstance(result, dict)
