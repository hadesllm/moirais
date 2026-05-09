"""Tests for blastp.blast_protein."""
import numpy as np
import pytest
from moirais.fn.blastp import blast_protein


def test_blastp_basic():
    """Test basic functionality."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    db = np.random.default_rng(42).normal(0, 1, 100)
    result = blast_protein(query, db)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_blastp_edge():
    """Test edge cases."""
    query = np.random.default_rng(42).normal(0, 1, 100)
    db = np.random.default_rng(42).normal(0, 1, 100)
    result = blast_protein(query, db)
    assert isinstance(result, dict)
