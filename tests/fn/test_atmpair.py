"""Tests for atmpair.atom_pair_fp."""
import numpy as np
import pytest
from morie.fn.atmpair import atom_pair_fp


def test_atmpair_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    max_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = atom_pair_fp(smiles, n_bits, max_dist)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_atmpair_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    max_dist = np.random.default_rng(42).normal(0, 1, 100)
    result = atom_pair_fp(smiles, n_bits, max_dist)
    assert isinstance(result, dict)
