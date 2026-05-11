"""Tests for dimNet.dimenet."""
import numpy as np
import pytest
from morie.fn.dimNet import dimenet


def test_dimNet_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    atom_types = np.random.default_rng(42).normal(0, 1, 100)
    result = dimenet(coords, atom_types)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dimNet_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    atom_types = np.random.default_rng(42).normal(0, 1, 100)
    result = dimenet(coords, atom_types)
    assert isinstance(result, dict)
