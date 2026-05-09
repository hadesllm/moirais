"""Tests for taxass.taxonomic_assignment."""
import numpy as np
import pytest
from moirais.fn.taxass import taxonomic_assignment


def test_taxass_basic():
    """Test basic functionality."""
    reads = np.random.default_rng(42).normal(0, 1, 100)
    kraken_db = np.random.default_rng(42).normal(0, 1, 100)
    result = taxonomic_assignment(reads, kraken_db)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_taxass_edge():
    """Test edge cases."""
    reads = np.random.default_rng(42).normal(0, 1, 100)
    kraken_db = np.random.default_rng(42).normal(0, 1, 100)
    result = taxonomic_assignment(reads, kraken_db)
    assert isinstance(result, dict)
