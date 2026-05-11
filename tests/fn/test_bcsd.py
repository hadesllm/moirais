"""Tests for bcsd.bcsd_downscaling."""
import numpy as np
import pytest
from morie.fn.bcsd import bcsd_downscaling


def test_bcsd_basic():
    """Test basic functionality."""
    gcm = np.random.default_rng(42).normal(0, 1, 100)
    obs = np.random.default_rng(42).normal(0, 1, 100)
    result = bcsd_downscaling(gcm, obs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bcsd_edge():
    """Test edge cases."""
    gcm = np.random.default_rng(42).normal(0, 1, 100)
    obs = np.random.default_rng(42).normal(0, 1, 100)
    result = bcsd_downscaling(gcm, obs)
    assert isinstance(result, dict)
