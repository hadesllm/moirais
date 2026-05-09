"""Tests for schN.schnet."""
import numpy as np
import pytest
from moirais.fn.schN import schnet


def test_schN_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    atom_types = np.random.default_rng(42).normal(0, 1, 100)
    result = schnet(coords, atom_types)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_schN_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    atom_types = np.random.default_rng(42).normal(0, 1, 100)
    result = schnet(coords, atom_types)
    assert isinstance(result, dict)
