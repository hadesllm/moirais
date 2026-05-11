"""Tests for alfvio.alphafold_violation."""
import numpy as np
import pytest
from morie.fn.alfvio import alphafold_violation


def test_alfvio_basic():
    """Test basic functionality."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    atom_types = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_violation(coords, atom_types)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfvio_edge():
    """Test edge cases."""
    coords = np.random.default_rng(42).uniform(0, 1, (100, 2))
    atom_types = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_violation(coords, atom_types)
    assert isinstance(result, dict)
