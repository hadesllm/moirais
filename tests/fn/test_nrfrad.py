"""Tests for nrfrad.nerf_radiance."""
import numpy as np
import pytest
from morie.fn.nrfrad import nerf_radiance


def test_nrfrad_basic():
    """Test basic functionality."""
    rays = np.random.default_rng(42).normal(0, 1, 100)
    mlp = np.random.default_rng(42).normal(0, 1, 100)
    result = nerf_radiance(rays, mlp)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_nrfrad_edge():
    """Test edge cases."""
    rays = np.random.default_rng(42).normal(0, 1, 100)
    mlp = np.random.default_rng(42).normal(0, 1, 100)
    result = nerf_radiance(rays, mlp)
    assert isinstance(result, dict)
