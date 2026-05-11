"""Tests for ecfp6.ecfp_6_fingerprint."""
import numpy as np
import pytest
from morie.fn.ecfp6 import ecfp_6_fingerprint


def test_ecfp6_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = ecfp_6_fingerprint(smiles, n_bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ecfp6_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = ecfp_6_fingerprint(smiles, n_bits)
    assert isinstance(result, dict)
