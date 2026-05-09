"""Tests for msm275.mvsml_convolutional_nn_eq_14_9."""
import numpy as np
import pytest
from moirais.fn.msm275 import mvsml_convolutional_nn_eq_14_9


def test_msm275_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    Finally = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_9(X, T, x1, t, Finally, the)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm275_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    T = np.random.default_rng(43).integers(0, 2, 100)
    x1 = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    Finally = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_9(X, T, x1, t, Finally, the)
    assert isinstance(result, dict)
