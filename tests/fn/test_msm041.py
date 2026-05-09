"""Tests for msm041.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from moirais.fn.msm041 import mvsml_linear_mixed_models_eq_5_4


def test_msm041_basic():
    """Test basic functionality."""
    rcov = np.random.default_rng(42).normal(0, 1, 100)
    vs = np.random.default_rng(42).normal(0, 1, 100)
    units = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    Basic = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(rcov, vs, units, data, dat_F, Basic)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm041_edge():
    """Test edge cases."""
    rcov = np.random.default_rng(42).normal(0, 1, 100)
    vs = np.random.default_rng(42).normal(0, 1, 100)
    units = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    Basic = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(rcov, vs, units, data, dat_F, Basic)
    assert isinstance(result, dict)
