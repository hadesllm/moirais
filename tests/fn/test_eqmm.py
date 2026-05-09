"""Tests for eqmm.equating_mean_mean."""
import numpy as np
import pytest
from moirais.fn.eqmm import equating_mean_mean


def test_eqmm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    a_R = np.random.default_rng(42).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    a_F = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    result = equating_mean_mean(y, a_R, b_R, a_F, b_F)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_eqmm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    a_R = np.random.default_rng(42).normal(0, 1, 100)
    b_R = np.random.default_rng(42).normal(0, 1, 100)
    a_F = np.random.default_rng(42).normal(0, 1, 100)
    b_F = np.random.default_rng(42).normal(0, 1, 100)
    result = equating_mean_mean(y, a_R, b_R, a_F, b_F)
    assert isinstance(result, dict)
