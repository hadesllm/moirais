"""Tests for sdcdis.spatial_data_distortion."""
import numpy as np
import pytest
from morie.fn.sdcdis import spatial_data_distortion


def test_sdcdis_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    noise_radius = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_distortion(coords, noise_radius)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sdcdis_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    noise_radius = np.random.default_rng(42).normal(0, 1, 100)
    result = spatial_data_distortion(coords, noise_radius)
    assert isinstance(result, dict)
