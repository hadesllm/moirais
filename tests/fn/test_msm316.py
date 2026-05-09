"""Tests for msm316.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from moirais.fn.msm316 import mvsml_general_eq_1_2


def test_msm316_basic():
    """Test basic functionality."""
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    dat_ls = np.random.default_rng(42).normal(0, 1, 100)
    head = np.random.default_rng(42).normal(0, 1, 100)
    Wavelengths = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    dat_W = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(dat_F, dat_ls, head, Wavelengths, data, dat_W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm316_edge():
    """Test edge cases."""
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    dat_ls = np.random.default_rng(42).normal(0, 1, 100)
    head = np.random.default_rng(42).normal(0, 1, 100)
    Wavelengths = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    dat_W = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(dat_F, dat_ls, head, Wavelengths, data, dat_W)
    assert isinstance(result, dict)
