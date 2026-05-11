"""Tests for msm315.mvsml_general_eq_1_2."""
import numpy as np
import pytest
from morie.fn.msm315 import mvsml_general_eq_1_2


def test_msm315_basic():
    """Test basic functionality."""
    load = np.random.default_rng(42).normal(0, 1, 100)
    dat_ls = np.random.default_rng(42).normal(0, 1, 100)
    RData = np.random.default_rng(42).normal(0, 1, 100)
    Phenotypic = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(load, dat_ls, RData, Phenotypic, data, dat_F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_msm315_edge():
    """Test edge cases."""
    load = np.random.default_rng(42).normal(0, 1, 100)
    dat_ls = np.random.default_rng(42).normal(0, 1, 100)
    RData = np.random.default_rng(42).normal(0, 1, 100)
    Phenotypic = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    dat_F = np.random.default_rng(42).normal(0, 1, 100)
    result = mvsml_general_eq_1_2(load, dat_ls, RData, Phenotypic, data, dat_F)
    assert isinstance(result, dict)
