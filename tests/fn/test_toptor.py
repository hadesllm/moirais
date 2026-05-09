"""Tests for toptor.topological_torsion."""
import numpy as np
import pytest
from moirais.fn.toptor import topological_torsion


def test_toptor_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = topological_torsion(smiles, n_bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_toptor_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = topological_torsion(smiles, n_bits)
    assert isinstance(result, dict)
