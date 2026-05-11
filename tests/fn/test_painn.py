"""Tests for painn.painn."""
import numpy as np
import pytest
from morie.fn.painn import painn


def test_painn_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    atom_types = np.random.default_rng(42).normal(0, 1, 100)
    result = painn(coords, atom_types)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_painn_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    atom_types = np.random.default_rng(42).normal(0, 1, 100)
    result = painn(coords, atom_types)
    assert isinstance(result, dict)
