"""Tests for smplqc.sample_qc."""
import numpy as np
import pytest
from moirais.fn.smplqc import sample_qc


def test_smplqc_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_qc(genotypes, filters)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_smplqc_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    filters = np.random.default_rng(42).normal(0, 1, 100)
    result = sample_qc(genotypes, filters)
    assert isinstance(result, dict)
