"""Tests for spmds.schabenberger_multidim_scaling."""
import numpy as np
import pytest
from moirais.fn.spmds import schabenberger_multidim_scaling


def test_spmds_basic():
    """Test basic functionality."""
    distance_matrix = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_multidim_scaling(distance_matrix, dim)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_spmds_edge():
    """Test edge cases."""
    distance_matrix = np.random.default_rng(42).normal(0, 1, 100)
    dim = np.random.default_rng(42).normal(0, 1, 100)
    result = schabenberger_multidim_scaling(distance_matrix, dim)
    assert isinstance(result, dict)
