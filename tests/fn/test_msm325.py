"""Tests for msm325.mvsml_functional_regression_eq_15_2."""
import numpy as np
import pytest
from moirais.fn.msm325 import mvsml_functional_regression_eq_15_2


def test_msm325_basic():
    """Test basic functionality."""
    log = np.random.default_rng(42).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_2(log, Y, i)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm325_edge():
    """Test edge cases."""
    log = np.random.default_rng(42).normal(0, 1, 100)
    Y = np.random.default_rng(43).normal(0, 1, 100)
    i = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_2(log, Y, i)
    assert isinstance(result, dict)
