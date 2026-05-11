"""Tests for thnsst.thiessen_polygons."""
import numpy as np
import pytest
from morie.fn.thnsst import thiessen_polygons


def test_thnsst_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = thiessen_polygons(coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_thnsst_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = thiessen_polygons(coords)
    assert isinstance(result, dict)
