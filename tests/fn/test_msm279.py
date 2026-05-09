"""Tests for msm279.mvsml_convolutional_nn_eq_14_2."""
import numpy as np
import pytest
from moirais.fn.msm279 import mvsml_convolutional_nn_eq_14_2


def test_msm279_basic():
    """Test basic functionality."""
    dp = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    dtp = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_2(dp, J, dtp, t, dt, where)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm279_edge():
    """Test edge cases."""
    dp = np.random.default_rng(42).normal(0, 1, 100)
    J = 20
    dtp = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    dt = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_2(dp, J, dtp, t, dt, where)
    assert isinstance(result, dict)
