"""Tests for msm052.mvsml_bayesian_regression_eq_6_4."""
import numpy as np
import pytest
from moirais.fn.msm052 import mvsml_bayesian_regression_eq_6_4


def test_msm052_basic():
    """Test basic functionality."""
    here = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    Nn = np.random.default_rng(42).normal(0, 1, 100)
    eg = np.random.default_rng(42).normal(0, 1, 100)
    eG = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(here, g, j, Nn, eg, eG)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm052_edge():
    """Test edge cases."""
    here = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    j = np.random.default_rng(42).normal(0, 1, 100)
    Nn = np.random.default_rng(42).normal(0, 1, 100)
    eg = np.random.default_rng(42).normal(0, 1, 100)
    eG = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_bayesian_regression_eq_6_4(here, g, j, Nn, eg, eG)
    assert isinstance(result, dict)
