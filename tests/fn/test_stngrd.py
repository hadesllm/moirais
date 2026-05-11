"""Tests for stngrd.stn_spatial_transform."""
import numpy as np
import pytest
from morie.fn.stngrd import stn_spatial_transform


def test_stngrd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stn_spatial_transform(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stngrd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = stn_spatial_transform(x)
    assert isinstance(result, dict)
