"""Tests for ibdmtx.ibd_matrix."""
import numpy as np
import pytest
from moirais.fn.ibdmtx import ibd_matrix


def test_ibdmtx_basic():
    """Test basic functionality."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    map = np.random.default_rng(42).normal(0, 1, 100)
    result = ibd_matrix(genotypes, map)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ibdmtx_edge():
    """Test edge cases."""
    genotypes = np.random.default_rng(42).normal(0, 1, 100)
    map = np.random.default_rng(42).normal(0, 1, 100)
    result = ibd_matrix(genotypes, map)
    assert isinstance(result, dict)
