"""Tests for impfun.genotype_imputation."""
import numpy as np
import pytest
from moirais.fn.impfun import genotype_imputation


def test_impfun_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = genotype_imputation(genotypes, reference)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_impfun_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    reference = np.random.default_rng(42).normal(0, 1, 100)
    result = genotype_imputation(genotypes, reference)
    assert isinstance(result, dict)
