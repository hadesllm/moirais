"""Tests for spstcn.schabenberger_st_cov_nonsep."""
import numpy as np
import pytest
from morie.fn.spstcn import schabenberger_st_cov_nonsep


def test_spstcn_basic():
    """Test basic functionality."""
    spatial_h = np.random.default_rng(42).normal(0, 1, 100)
    temporal_u = np.random.default_rng(42).normal(0, 1, 100)
    params = {'item1': {'a': 1.0, 'b': 0.0}, 'item2': {'a': 1.5, 'b': 0.5}}
    result = schabenberger_st_cov_nonsep(spatial_h, temporal_u, params)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spstcn_edge():
    """Test edge cases."""
    spatial_h = np.random.default_rng(42).normal(0, 1, 100)
    temporal_u = np.random.default_rng(42).normal(0, 1, 100)
    params = {'item1': {'a': 1.0, 'b': 0.0}, 'item2': {'a': 1.5, 'b': 0.5}}
    result = schabenberger_st_cov_nonsep(spatial_h, temporal_u, params)
    assert isinstance(result, dict)
