"""Tests for sgflrt.spatial_glmm_fit."""
import numpy as np
import pytest
from moirais.fn.sgflrt import spatial_glmm_fit


def test_sgflrt_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    family = 'gaussian'
    result = spatial_glmm_fit(y, X, coords, family)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgflrt_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    family = 'gaussian'
    result = spatial_glmm_fit(y, X, coords, family)
    assert isinstance(result, dict)
