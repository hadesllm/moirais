"""Tests for avalon.avalon_fingerprint."""
import numpy as np
import pytest
from morie.fn.avalon import avalon_fingerprint


def test_avalon_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = avalon_fingerprint(smiles, n_bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_avalon_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = avalon_fingerprint(smiles, n_bits)
    assert isinstance(result, dict)
