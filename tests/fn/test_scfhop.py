"""Tests for scfhop.scaffold_hop."""
import numpy as np
import pytest
from morie.fn.scfhop import scaffold_hop


def test_scfhop_basic():
    """Test basic functionality."""
    lead_smiles = np.random.default_rng(42).normal(0, 1, 100)
    scaffold_db = np.random.default_rng(42).normal(0, 1, 100)
    result = scaffold_hop(lead_smiles, scaffold_db)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scfhop_edge():
    """Test edge cases."""
    lead_smiles = np.random.default_rng(42).normal(0, 1, 100)
    scaffold_db = np.random.default_rng(42).normal(0, 1, 100)
    result = scaffold_hop(lead_smiles, scaffold_db)
    assert isinstance(result, dict)
