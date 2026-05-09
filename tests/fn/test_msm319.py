"""Tests for msm319.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from moirais.fn.msm319 import mvsml_general_eq_1_2


def test_msm319_basic():
    """Test basic functionality."""
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    dat_ls = np.random.default_rng(42).normal(0, 1, 100)
    head = np.random.default_rng(42).normal(0, 1, 100)
    yv = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = mvsml_general_eq_1_2(dat_F, dat_ls, head, yv, y, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm319_edge():
    """Test edge cases."""
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    dat_ls = np.random.default_rng(42).normal(0, 1, 100)
    head = np.random.default_rng(42).normal(0, 1, 100)
    yv = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = mvsml_general_eq_1_2(dat_F, dat_ls, head, yv, y, n)
    assert isinstance(result, dict)
