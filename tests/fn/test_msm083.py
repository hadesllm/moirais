"""Tests for msm083.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from morie.fn.msm083 import mvsml_general_eq_1_2


def test_msm083_basic():
    """Test basic functionality."""
    Appendix = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    Code = np.random.default_rng(42).normal(0, 1, 100)
    Example = np.random.default_rng(42).normal(0, 1, 100)
    rm = np.random.default_rng(42).normal(0, 1, 100)
    list = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(Appendix, R, Code, Example, rm, list)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm083_edge():
    """Test edge cases."""
    Appendix = np.random.default_rng(42).normal(0, 1, 100)
    R = np.random.default_rng(42).normal(0, 1, 100)
    Code = np.random.default_rng(42).normal(0, 1, 100)
    Example = np.random.default_rng(42).normal(0, 1, 100)
    rm = np.random.default_rng(42).normal(0, 1, 100)
    list = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(Appendix, R, Code, Example, rm, list)
    assert isinstance(result, dict)
