"""Tests for msm283.mvsml_convolutional_nn_eq_14_12."""
import numpy as np
import pytest
from morie.fn.msm283 import mvsml_convolutional_nn_eq_14_12


def test_msm283_basic():
    """Test basic functionality."""
    be = np.random.default_rng(42).normal(0, 1, 100)
    written = np.random.default_rng(42).normal(0, 1, 100)
    j2 = np.random.default_rng(42).normal(0, 1, 100)
    TD = np.random.default_rng(42).normal(0, 1, 100)
    SSE = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(be, written, j2, TD, SSE, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm283_edge():
    """Test edge cases."""
    be = np.random.default_rng(42).normal(0, 1, 100)
    written = np.random.default_rng(42).normal(0, 1, 100)
    j2 = np.random.default_rng(42).normal(0, 1, 100)
    TD = np.random.default_rng(42).normal(0, 1, 100)
    SSE = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_12(be, written, j2, TD, SSE, y)
    assert isinstance(result, dict)
