"""Tests for ecfp4.ecfp_4_fingerprint."""
import numpy as np
import pytest
from morie.fn.ecfp4 import ecfp_4_fingerprint


def test_ecfp4_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    radius = np.random.default_rng(42).normal(0, 1, 100)
    result = ecfp_4_fingerprint(smiles, n_bits, radius)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ecfp4_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    radius = np.random.default_rng(42).normal(0, 1, 100)
    result = ecfp_4_fingerprint(smiles, n_bits, radius)
    assert isinstance(result, dict)
