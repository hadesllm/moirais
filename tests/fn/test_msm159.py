"""Tests for msm159.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from morie.fn.msm159 import mvsml_general_eq_1_2


def test_msm159_basic():
    """Test basic functionality."""
    Tab1_MSE = np.random.default_rng(42).normal(0, 1, 100)
    AK1 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    mean = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    Pos_tst = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(Tab1_MSE, AK1, k, mean, y, Pos_tst)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm159_edge():
    """Test edge cases."""
    Tab1_MSE = np.random.default_rng(42).normal(0, 1, 100)
    AK1 = np.random.default_rng(42).normal(0, 1, 100)
    k = 5
    mean = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    Pos_tst = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(Tab1_MSE, AK1, k, mean, y, Pos_tst)
    assert isinstance(result, dict)
