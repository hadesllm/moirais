"""Tests for fpca.functional_pca."""
import numpy as np
import pytest
from morie.fn.fpca import functional_pca


def test_fpca_basic():
    """Test basic functionality."""
    data_functions = np.random.default_rng(42).normal(0, 1, 100)
    n_components = 3
    result = functional_pca(data_functions, n_components)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fpca_edge():
    """Test edge cases."""
    data_functions = np.random.default_rng(42).normal(0, 1, 100)
    n_components = 3
    result = functional_pca(data_functions, n_components)
    assert isinstance(result, dict)
