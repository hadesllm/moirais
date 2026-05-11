"""Tests for rotbnd.rotatable_bond_count."""
import numpy as np
import pytest
from morie.fn.rotbnd import rotatable_bond_count


def test_rotbnd_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = rotatable_bond_count(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rotbnd_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = rotatable_bond_count(smiles)
    assert isinstance(result, dict)
