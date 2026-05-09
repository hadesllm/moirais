"""Tests for blstn.blast_nucleotide."""
import numpy as np
import pytest
from moirais.fn.blstn import blast_nucleotide


def test_blstn_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    db = np.random.default_rng(42).normal(0, 1, 100)
    result = blast_nucleotide(query, db)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blstn_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    db = np.random.default_rng(42).normal(0, 1, 100)
    result = blast_nucleotide(query, db)
    assert isinstance(result, dict)
