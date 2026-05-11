"""Tests for adseqs.admixture_seq."""
import numpy as np
import pytest
from morie.fn.adseqs import admixture_seq


def test_adseqs_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = admixture_seq(genotypes, K)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adseqs_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    K = np.eye(10) + 0.1 * np.random.default_rng(43).normal(0, 1, (10, 10))
    result = admixture_seq(genotypes, K)
    assert isinstance(result, dict)
