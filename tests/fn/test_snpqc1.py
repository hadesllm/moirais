"""Tests for snpqc1.snp_qc."""
import numpy as np
import pytest
from morie.fn.snpqc1 import snp_qc


def test_snpqc1_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = snp_qc(genotypes, filters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snpqc1_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = snp_qc(genotypes, filters)
    assert isinstance(result, dict)
