"""Tests for spcllm.spatial_cluster_lisa."""
import numpy as np
import pytest
from moirais.fn.spcllm import spatial_cluster_lisa


def test_spcllm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = spatial_cluster_lisa(x, W, alpha)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcllm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    alpha = 0.05
    result = spatial_cluster_lisa(x, W, alpha)
    assert isinstance(result, dict)
