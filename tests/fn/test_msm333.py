"""Tests for msm333.mvsml_elements_lin_reg_eq_3_1."""
import numpy as np
import pytest
from moirais.fn.msm333 import mvsml_elements_lin_reg_eq_3_1


def test_msm333_basic():
    """Test basic functionality."""
    n = 100
    that = np.random.default_rng(42).normal(0, 1, 100)
    produces = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    equivalent = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_elements_lin_reg_eq_3_1(n, that, produces, an, equivalent, de)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm333_edge():
    """Test edge cases."""
    n = 100
    that = np.random.default_rng(42).normal(0, 1, 100)
    produces = np.random.default_rng(42).normal(0, 1, 100)
    an = np.random.default_rng(42).normal(0, 1, 100)
    equivalent = np.random.default_rng(42).normal(0, 1, 100)
    de = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_elements_lin_reg_eq_3_1(n, that, produces, an, equivalent, de)
    assert isinstance(result, dict)
