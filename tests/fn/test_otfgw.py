"""Tests for otfgw.ot_fused_gromov_wasserstein."""
import numpy as np
import pytest
from moirais.fn.otfgw import ot_fused_gromov_wasserstein


def test_otfgw_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Cx = np.random.default_rng(42).normal(0, 1, 100)
    Cy = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_fused_gromov_wasserstein(M, Cx, Cy, a, b, alpha, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otfgw_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    Cx = np.random.default_rng(42).normal(0, 1, 100)
    Cy = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_fused_gromov_wasserstein(M, Cx, Cy, a, b, alpha, max_iter)
    assert isinstance(result, dict)
