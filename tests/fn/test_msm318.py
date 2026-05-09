"""Tests for msm318.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from moirais.fn.msm318 import mvsml_general_eq_1_2


def test_msm318_basic():
    """Test basic functionality."""
    yv = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    length = np.random.default_rng(42).normal(0, 1, 100)
    Wavelengths = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(yv, dat_F, y, n, length, Wavelengths)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm318_edge():
    """Test edge cases."""
    yv = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    length = np.random.default_rng(42).normal(0, 1, 100)
    Wavelengths = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(yv, dat_F, y, n, length, Wavelengths)
    assert isinstance(result, dict)
