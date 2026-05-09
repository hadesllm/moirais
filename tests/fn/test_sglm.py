"""Tests for sglm.spatial_glm."""
import numpy as np
import pytest
from moirais.fn.sglm import spatial_glm


def test_sglm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_glm(x, y, coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sglm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = spatial_glm(x, y, coords)
    assert isinstance(result, dict)
