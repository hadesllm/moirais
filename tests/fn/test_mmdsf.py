"""Tests for mmdsf.metric_mds_torgerson."""
import numpy as np
import pytest
from moirais.fn.mmdsf import metric_mds_torgerson


def test_mmdsf_basic():
    """Test basic functionality."""
    D_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    n_dims = 2
    result = metric_mds_torgerson(D_matrix, n_dims)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mmdsf_edge():
    """Test edge cases."""
    D_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    n_dims = 2
    result = metric_mds_torgerson(D_matrix, n_dims)
    assert isinstance(result, dict)
