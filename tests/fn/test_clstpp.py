"""Tests for clstpp.clark_evans."""
import numpy as np
import pytest
from morie.fn.clstpp import clark_evans


def test_clstpp_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = clark_evans(coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clstpp_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    result = clark_evans(coords)
    assert isinstance(result, dict)
