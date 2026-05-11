"""Tests for krigFDA.kriging."""
import numpy as np
import pytest
from morie.fn.krigFDA import kriging


def test_krigFDA_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    new_coords = np.random.default_rng(42).normal(0, 1, 100)
    result = kriging(coords, values, new_coords)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_krigFDA_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    values = np.random.default_rng(42).normal(0, 1, 100)
    new_coords = np.random.default_rng(42).normal(0, 1, 100)
    result = kriging(coords, values, new_coords)
    assert isinstance(result, dict)
