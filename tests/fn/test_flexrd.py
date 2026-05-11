"""Tests for flexrd.flexible_receptor_dock."""
import numpy as np
import pytest
from morie.fn.flexrd import flexible_receptor_dock


def test_flexrd_basic():
    """Test basic functionality."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    flex_residues = np.random.default_rng(42).normal(0, 1, 100)
    result = flexible_receptor_dock(receptor, ligand, flex_residues)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_flexrd_edge():
    """Test edge cases."""
    receptor = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    flex_residues = np.random.default_rng(42).normal(0, 1, 100)
    result = flexible_receptor_dock(receptor, ligand, flex_residues)
    assert isinstance(result, dict)
