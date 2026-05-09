"""Tests for msm258.mvsml_elements_lin_reg_eq_3_5."""
import numpy as np
import pytest
from moirais.fn.msm258 import mvsml_elements_lin_reg_eq_3_5


def test_msm258_basic():
    """Test basic functionality."""
    colnames = np.random.default_rng(42).normal(0, 1, 100)
    results_i = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    Observed = np.random.default_rng(42).normal(0, 1, 100)
    Predicted = np.random.default_rng(42).normal(0, 1, 100)
    Trait = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_elements_lin_reg_eq_3_5(colnames, results_i, c, Observed, Predicted, Trait)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm258_edge():
    """Test edge cases."""
    colnames = np.random.default_rng(42).normal(0, 1, 100)
    results_i = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    Observed = np.random.default_rng(42).normal(0, 1, 100)
    Predicted = np.random.default_rng(42).normal(0, 1, 100)
    Trait = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_elements_lin_reg_eq_3_5(colnames, results_i, c, Observed, Predicted, Trait)
    assert isinstance(result, dict)
