"""Tests for msm289.mvsml_convolutional_nn_eq_14_12."""
import numpy as np
import pytest
from moirais.fn.msm289 import mvsml_convolutional_nn_eq_14_12


def test_msm289_basic():
    """Test basic functionality."""
    previous = np.random.default_rng(42).normal(0, 1, 100)
    section = np.random.default_rng(42).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    partitions = np.random.default_rng(42).normal(0, 1, 100)
    were = np.random.default_rng(42).normal(0, 1, 100)
    used = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(previous, section, random, partitions, were, used)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm289_edge():
    """Test edge cases."""
    previous = np.random.default_rng(42).normal(0, 1, 100)
    section = np.random.default_rng(42).normal(0, 1, 100)
    random = np.random.default_rng(42).normal(0, 1, 100)
    partitions = np.random.default_rng(42).normal(0, 1, 100)
    were = np.random.default_rng(42).normal(0, 1, 100)
    used = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(previous, section, random, partitions, were, used)
    assert isinstance(result, dict)
