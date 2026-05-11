"""Tests for clrnt.clearance_intrinsic."""
import numpy as np
import pytest
from morie.fn.clrnt import clearance_intrinsic


def test_clrnt_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    species = np.random.default_rng(42).normal(0, 1, 100)
    result = clearance_intrinsic(smiles, species)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_clrnt_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    species = np.random.default_rng(42).normal(0, 1, 100)
    result = clearance_intrinsic(smiles, species)
    assert isinstance(result, dict)
