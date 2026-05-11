"""Tests for msm310.mvsml_convolutional_nn_eq_14_14."""
import numpy as np
import pytest
from morie.fn.msm310 import mvsml_convolutional_nn_eq_14_14


def test_msm310_basic():
    """Test basic functionality."""
    no = np.random.default_rng(42).normal(0, 1, 100)
    interaction = np.random.default_rng(42).normal(0, 1, 100)
    term = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    We = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(no, interaction, term, model, We, can)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm310_edge():
    """Test edge cases."""
    no = np.random.default_rng(42).normal(0, 1, 100)
    interaction = np.random.default_rng(42).normal(0, 1, 100)
    term = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    We = np.random.default_rng(42).normal(0, 1, 100)
    can = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_14(no, interaction, term, model, We, can)
    assert isinstance(result, dict)
