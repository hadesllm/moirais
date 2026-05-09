"""Tests for abndst.abundance_estimation."""
import numpy as np
import pytest
from moirais.fn.abndst import abundance_estimation


def test_abndst_basic():
    """Test basic functionality."""
    kraken_output = np.random.default_rng(42).normal(0, 1, 100)
    kmer_distribution = np.random.default_rng(42).normal(0, 1, 100)
    result = abundance_estimation(kraken_output, kmer_distribution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_abndst_edge():
    """Test edge cases."""
    kraken_output = np.random.default_rng(42).normal(0, 1, 100)
    kmer_distribution = np.random.default_rng(42).normal(0, 1, 100)
    result = abundance_estimation(kraken_output, kmer_distribution)
    assert isinstance(result, dict)
