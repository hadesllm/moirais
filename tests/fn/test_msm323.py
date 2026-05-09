"""Tests for msm323.mvsml_functional_regression_eq_15_1."""
import numpy as np
import pytest
from moirais.fn.msm323 import mvsml_functional_regression_eq_15_1


def test_msm323_basic():
    """Test basic functionality."""
    Mathlouthi = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    al = np.random.default_rng(42).normal(0, 1, 100)
    through = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    given = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_1(Mathlouthi, et, al, through, are, given)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm323_edge():
    """Test edge cases."""
    Mathlouthi = np.random.default_rng(42).normal(0, 1, 100)
    et = np.random.default_rng(42).normal(0, 1, 100)
    al = np.random.default_rng(42).normal(0, 1, 100)
    through = np.random.default_rng(42).normal(0, 1, 100)
    are = np.random.default_rng(42).normal(0, 1, 100)
    given = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_functional_regression_eq_15_1(Mathlouthi, et, al, through, are, given)
    assert isinstance(result, dict)
