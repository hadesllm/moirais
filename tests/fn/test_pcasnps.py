"""Tests for pcasnps.pca_snps."""
import numpy as np
import pytest
from morie.fn.pcasnps import pca_snps


def test_pcasnps_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    n_components = 3
    result = pca_snps(genotypes, n_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pcasnps_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    n_components = 3
    result = pca_snps(genotypes, n_components)
    assert isinstance(result, dict)
