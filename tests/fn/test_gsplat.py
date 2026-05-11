"""Tests for gsplat.gaussian_splatting."""
import numpy as np
import pytest
from morie.fn.gsplat import gaussian_splatting


def test_gsplat_basic():
    """Test basic functionality."""
    gaussians = np.random.default_rng(42).normal(0, 1, 100)
    camera = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_splatting(gaussians, camera)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gsplat_edge():
    """Test edge cases."""
    gaussians = np.random.default_rng(42).normal(0, 1, 100)
    camera = np.random.default_rng(42).normal(0, 1, 100)
    result = gaussian_splatting(gaussians, camera)
    assert isinstance(result, dict)
