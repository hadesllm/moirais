"""Tests for grfmp.geron_feature_map_dim."""
import numpy as np
import pytest
from morie.fn.grfmp import geron_feature_map_dim


def test_grfmp_basic():
    """Test basic functionality."""
    H_out = np.random.default_rng(42).normal(0, 1, 100)
    W_out = np.random.default_rng(42).normal(0, 1, 100)
    C_out = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_feature_map_dim(H_out, W_out, C_out)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_grfmp_edge():
    """Test edge cases."""
    H_out = np.random.default_rng(42).normal(0, 1, 100)
    W_out = np.random.default_rng(42).normal(0, 1, 100)
    C_out = np.random.default_rng(42).normal(0, 1, 100)
    result = geron_feature_map_dim(H_out, W_out, C_out)
    assert isinstance(result, dict)
