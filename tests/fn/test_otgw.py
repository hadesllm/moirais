"""Tests for otgw.ot_gromov_wasserstein."""
import numpy as np
import pytest
from moirais.fn.otgw import ot_gromov_wasserstein


def test_otgw_basic():
    """Test basic functionality."""
    Cx = np.random.default_rng(42).normal(0, 1, 100)
    Cy = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_gromov_wasserstein(Cx, Cy, a, b, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otgw_edge():
    """Test edge cases."""
    Cx = np.random.default_rng(42).normal(0, 1, 100)
    Cy = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_gromov_wasserstein(Cx, Cy, a, b, max_iter)
    assert isinstance(result, dict)
