"""Tests for msm037.mvsml_linear_mixed_models_eq_5_4."""
import numpy as np
import pytest
from moirais.fn.msm037 import mvsml_linear_mixed_models_eq_5_4


def test_msm037_basic():
    """Test basic functionality."""
    dat_M = np.random.default_rng(42).normal(0, 1, 100)
    scale = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    tcrossprod = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(dat_M, scale, G, tcrossprod, dim, dat_F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm037_edge():
    """Test edge cases."""
    dat_M = np.random.default_rng(42).normal(0, 1, 100)
    scale = np.random.default_rng(42).normal(0, 1, 100)
    G = np.eye(10)
    tcrossprod = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_linear_mixed_models_eq_5_4(dat_M, scale, G, tcrossprod, dim, dat_F)
    assert isinstance(result, dict)
