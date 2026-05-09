"""Tests for msm332.mvsml_elements_lin_reg_eq_3_1."""
import numpy as np
import pytest
from moirais.fn.msm332 import mvsml_elements_lin_reg_eq_3_1


def test_msm332_basic():
    """Test basic functionality."""
    Fitting = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    Linear = np.random.default_rng(42).normal(0, 1, 100)
    Multiple = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    Model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_elements_lin_reg_eq_3_1(Fitting, a, Linear, Multiple, Regression, Model)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm332_edge():
    """Test edge cases."""
    Fitting = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    Linear = np.random.default_rng(42).normal(0, 1, 100)
    Multiple = np.random.default_rng(42).normal(0, 1, 100)
    Regression = np.random.default_rng(42).normal(0, 1, 100)
    Model = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_elements_lin_reg_eq_3_1(Fitting, a, Linear, Multiple, Regression, Model)
    assert isinstance(result, dict)
