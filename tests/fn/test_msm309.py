"""Tests for msm309.mvsml_convolutional_nn_eq_14_14."""
import numpy as np
import pytest
from moirais.fn.msm309 import mvsml_convolutional_nn_eq_14_14


def test_msm309_basic():
    """Test basic functionality."""
    ance = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    no = np.random.default_rng(42).normal(0, 1, 100)
    interaction = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(ance, of, the, model, no, interaction)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm309_edge():
    """Test edge cases."""
    ance = np.random.default_rng(42).normal(0, 1, 100)
    of = np.random.default_rng(42).normal(0, 1, 100)
    the = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    no = np.random.default_rng(42).normal(0, 1, 100)
    interaction = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(ance, of, the, model, no, interaction)
    assert isinstance(result, dict)
