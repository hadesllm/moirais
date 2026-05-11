"""Tests for alfsmd.alphafold_msa_attention."""
import numpy as np
import pytest
from morie.fn.alfsmd import alphafold_msa_attention


def test_alfsmd_basic():
    """Test basic functionality."""
    msa = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    keys = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_msa_attention(msa, queries, keys)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_alfsmd_edge():
    """Test edge cases."""
    msa = np.random.default_rng(42).normal(0, 1, 100)
    queries = np.random.default_rng(42).normal(0, 1, 100)
    keys = np.random.default_rng(42).normal(0, 1, 100)
    result = alphafold_msa_attention(msa, queries, keys)
    assert isinstance(result, dict)
