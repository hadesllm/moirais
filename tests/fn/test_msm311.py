"""Tests for msm311.mvsml_convolutional_nn_eq_14_13."""
import numpy as np
import pytest
from moirais.fn.msm311 import mvsml_convolutional_nn_eq_14_13


def test_msm311_basic():
    """Test basic functionality."""
    l = np.random.default_rng(42).normal(0, 1, 100)
    covariate = np.random.default_rng(42).normal(0, 1, 100)
    both = np.random.default_rng(42).normal(0, 1, 100)
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(l, covariate, both, Bayesian, models, PBFR)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm311_edge():
    """Test edge cases."""
    l = np.random.default_rng(42).normal(0, 1, 100)
    covariate = np.random.default_rng(42).normal(0, 1, 100)
    both = np.random.default_rng(42).normal(0, 1, 100)
    Bayesian = np.random.default_rng(42).normal(0, 1, 100)
    models = np.random.default_rng(42).normal(0, 1, 100)
    PBFR = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_convolutional_nn_eq_14_13(l, covariate, both, Bayesian, models, PBFR)
    assert isinstance(result, dict)
