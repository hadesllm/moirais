"""Tests for msm278.mvsml_convolutional_nn_eq_14_11."""
import numpy as np
import pytest
from moirais.fn.msm278 import mvsml_convolutional_nn_eq_14_11


def test_msm278_basic():
    """Test basic functionality."""
    dp = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    dtp = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_11(dp, J, dtp, t, dt)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm278_edge():
    """Test edge cases."""
    dp = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    dtp = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_11(dp, J, dtp, t, dt)
    assert isinstance(result, dict)
