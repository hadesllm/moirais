"""Tests for msm286.mvsml_convolutional_nn_eq_14_10."""
import numpy as np
import pytest
from morie.fn.msm286 import mvsml_convolutional_nn_eq_14_10


def test_msm286_basic():
    """Test basic functionality."""
    smoothed = np.random.default_rng(42).normal(0, 1, 100)
    solution = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_10(smoothed, solution, of, t, can, be)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm286_edge():
    """Test edge cases."""
    smoothed = np.random.default_rng(42).normal(0, 1, 100)
    solution = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    be = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_10(smoothed, solution, of, t, can, be)
    assert isinstance(result, dict)
