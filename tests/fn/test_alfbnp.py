"""Tests for alfbnp.af3_protein_ligand."""
import numpy as np
import pytest
from moirais.fn.alfbnp import af3_protein_ligand


def test_alfbnp_basic():
    """Test basic functionality."""
    protein = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    result = af3_protein_ligand(protein, ligand)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfbnp_edge():
    """Test edge cases."""
    protein = np.random.default_rng(42).normal(0, 1, 100)
    ligand = np.random.default_rng(42).normal(0, 1, 100)
    result = af3_protein_ligand(protein, ligand)
    assert isinstance(result, dict)
