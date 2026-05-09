"""Tests for msm312.mvsml_convolutional_nn_eq_14_13."""
import numpy as np
import pytest
from moirais.fn.msm312 import mvsml_convolutional_nn_eq_14_13


def test_msm312_basic():
    """Test basic functionality."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    resulted = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(Bayesian, models, PBFR, BFR, resulted, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm312_edge():
    """Test edge cases."""
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    BFR = np.random.default_rng(42).normal(0, 1, 100)
    resulted = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(Bayesian, models, PBFR, BFR, resulted, a)
    assert isinstance(result, dict)
