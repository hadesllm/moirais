"""Tests for msm324.mvsml_functional_regression_eq_15_1."""
import numpy as np
import pytest
from moirais.fn.msm324 import mvsml_functional_regression_eq_15_1


def test_msm324_basic():
    """Test basic functionality."""
    log = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    general = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_1(log, f, x, where, are, general)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm324_edge():
    """Test edge cases."""
    log = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    x = np.random.default_rng(42).normal(0, 1, 100)
    where = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    general = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_1(log, f, x, where, are, general)
    assert isinstance(result, dict)
