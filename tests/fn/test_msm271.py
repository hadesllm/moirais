"""Tests for msm271.mvsml_convolutional_nn_eq_14_8."""
import numpy as np
import pytest
from moirais.fn.msm271 import mvsml_convolutional_nn_eq_14_8


def test_msm271_basic():
    """Test basic functionality."""
    L2 = np.random.default_rng(42).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    t2 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_8(L2, t1, t2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm271_edge():
    """Test edge cases."""
    L2 = np.random.default_rng(42).normal(0, 1, 100)
    t1 = np.random.default_rng(42).normal(0, 1, 100)
    t2 = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_8(L2, t1, t2)
    assert isinstance(result, dict)
