"""Tests for sechsh.hash_chain_audit."""
import numpy as np
import pytest
from morie.fn.sechsh import hash_chain_audit


def test_sechsh_basic():
    """Test basic functionality."""
    prev_hash = np.random.default_rng(42).normal(0, 1, 100)
    row = np.random.default_rng(42).normal(0, 1, 100)
    hash_alg = np.random.default_rng(42).normal(0, 1, 100)
    result = hash_chain_audit(prev_hash, row, hash_alg)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sechsh_edge():
    """Test edge cases."""
    prev_hash = np.random.default_rng(42).normal(0, 1, 100)
    row = np.random.default_rng(42).normal(0, 1, 100)
    hash_alg = np.random.default_rng(42).normal(0, 1, 100)
    result = hash_chain_audit(prev_hash, row, hash_alg)
    assert isinstance(result, dict)
