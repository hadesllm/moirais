"""Tests for rdkfp.rdkit_path_fp."""
import numpy as np
import pytest
from morie.fn.rdkfp import rdkit_path_fp


def test_rdkfp_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    min_path = '/tmp/morie_test'
    max_path = '/tmp/morie_test'
    result = rdkit_path_fp(smiles, n_bits, min_path, max_path)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rdkfp_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    min_path = '/tmp/morie_test'
    max_path = '/tmp/morie_test'
    result = rdkit_path_fp(smiles, n_bits, min_path, max_path)
    assert isinstance(result, dict)
