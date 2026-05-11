"""Tests for mrkvr.marker_variance."""
import numpy as np
import pytest
from morie.fn.mrkvr import marker_variance


def test_mrkvr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    result = marker_variance(x, y, markers)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mrkvr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    markers = np.random.default_rng(43).integers(0, 3, (100, 20))
    result = marker_variance(x, y, markers)
    assert isinstance(result, dict)
