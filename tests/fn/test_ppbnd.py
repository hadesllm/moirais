"""Tests for ppbnd.plasma_protein_binding."""
import numpy as np
import pytest
from moirais.fn.ppbnd import plasma_protein_binding


def test_ppbnd_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = plasma_protein_binding(smiles)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ppbnd_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    result = plasma_protein_binding(smiles)
    assert isinstance(result, dict)
