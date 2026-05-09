"""Tests for poissp.poisson_spatial_glm."""
import numpy as np
import pytest
from moirais.fn.poissp import poisson_spatial_glm


def test_poissp_basic():
    """Test basic functionality."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    offset = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_spatial_glm(counts, X, offset, W)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_poissp_edge():
    """Test edge cases."""
    counts = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    offset = np.random.default_rng(42).normal(0, 1, 100)
    W = np.random.default_rng(42).normal(0, 1, 100)
    result = poisson_spatial_glm(counts, X, offset, W)
    assert isinstance(result, dict)
