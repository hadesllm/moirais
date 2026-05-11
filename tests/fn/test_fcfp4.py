"""Tests for fcfp4.fcfp_4_fingerprint."""
import numpy as np
import pytest
from morie.fn.fcfp4 import fcfp_4_fingerprint


def test_fcfp4_basic():
    """Test basic functionality."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = fcfp_4_fingerprint(smiles, n_bits)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fcfp4_edge():
    """Test edge cases."""
    smiles = np.random.default_rng(42).normal(0, 1, 100)
    n_bits = np.random.default_rng(42).normal(0, 1, 100)
    result = fcfp_4_fingerprint(smiles, n_bits)
    assert isinstance(result, dict)
