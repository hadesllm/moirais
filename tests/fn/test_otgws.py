"""Tests for otgws.ot_gromov_sinkhorn."""
import numpy as np
import pytest
from morie.fn.otgws import ot_gromov_sinkhorn


def test_otgws_basic():
    """Test basic functionality."""
    Cx = np.random.default_rng(42).normal(0, 1, 100)
    Cy = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_gromov_sinkhorn(Cx, Cy, a, b, epsilon, max_iter)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otgws_edge():
    """Test edge cases."""
    Cx = np.random.default_rng(42).normal(0, 1, 100)
    Cy = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    epsilon = 1e-6
    max_iter = np.random.default_rng(42).normal(0, 1, 100)
    result = ot_gromov_sinkhorn(Cx, Cy, a, b, epsilon, max_iter)
    assert isinstance(result, dict)
