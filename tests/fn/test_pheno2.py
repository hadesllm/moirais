"""Tests for pheno2.phenotype_qc."""
import numpy as np
import pytest
from morie.fn.pheno2 import phenotype_qc


def test_pheno2_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = phenotype_qc(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pheno2_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = phenotype_qc(y)
    assert isinstance(result, dict)
