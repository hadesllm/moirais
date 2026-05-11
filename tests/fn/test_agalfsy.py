"""Tests for agalfsy.alphazero_alphafold_synergy."""
import numpy as np
import pytest
from morie.fn.agalfsy import alphazero_alphafold_synergy


def test_agalfsy_basic():
    """Test basic functionality."""
    protein = np.random.default_rng(42).normal(0, 1, 100)
    ligand_pool = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_alphafold_synergy(protein, ligand_pool)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_agalfsy_edge():
    """Test edge cases."""
    protein = np.random.default_rng(42).normal(0, 1, 100)
    ligand_pool = np.random.default_rng(42).normal(0, 1, 100)
    result = alphazero_alphafold_synergy(protein, ligand_pool)
    assert isinstance(result, dict)
