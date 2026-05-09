"""Tests for msm269.mvsml_convolutional_nn_eq_14_6."""
import numpy as np
import pytest
from moirais.fn.msm269 import mvsml_convolutional_nn_eq_14_6


def test_msm269_basic():
    """Test basic functionality."""
    uously = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    Usually = np.random.default_rng(42).normal(0, 1, 100)
    it = np.random.default_rng(42).normal(0, 1, 100)
    only = np.random.default_rng(42).normal(0, 1, 100)
    measured = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_6(uously, observed, Usually, it, only, measured)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm269_edge():
    """Test edge cases."""
    uously = np.random.default_rng(42).normal(0, 1, 100)
    observed = np.random.default_rng(42).normal(0, 1, 100)
    Usually = np.random.default_rng(42).normal(0, 1, 100)
    it = np.random.default_rng(42).normal(0, 1, 100)
    only = np.random.default_rng(42).normal(0, 1, 100)
    measured = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_6(uously, observed, Usually, it, only, measured)
    assert isinstance(result, dict)
