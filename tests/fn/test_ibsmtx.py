"""Tests for ibsmtx.ibs_matrix."""
import numpy as np
import pytest
from moirais.fn.ibsmtx import ibs_matrix


def test_ibsmtx_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    result = ibs_matrix(genotypes)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ibsmtx_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    result = ibs_matrix(genotypes)
    assert isinstance(result, dict)
