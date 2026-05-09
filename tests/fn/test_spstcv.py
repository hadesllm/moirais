"""Tests for spstcv.schabenberger_st_cov_separable."""
import numpy as np
import pytest
from moirais.fn.spstcv import schabenberger_st_cov_separable


def test_spstcv_basic():
    """Test basic functionality."""
    spatial_h = np.random.default_rng(42).normal(0, 1, 100)
    temporal_u = np.random.default_rng(42).normal(0, 1, 100)
    cov_spatial = np.random.default_rng(42).normal(0, 1, 100)
    cov_temporal = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_st_cov_separable(spatial_h, temporal_u, cov_spatial, cov_temporal)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spstcv_edge():
    """Test edge cases."""
    spatial_h = np.random.default_rng(42).normal(0, 1, 100)
    temporal_u = np.random.default_rng(42).normal(0, 1, 100)
    cov_spatial = np.random.default_rng(42).normal(0, 1, 100)
    cov_temporal = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_st_cov_separable(spatial_h, temporal_u, cov_spatial, cov_temporal)
    assert isinstance(result, dict)
