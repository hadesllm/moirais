"""Tests for sgtnlap.sgt_normalised_laplacian."""
import numpy as np
import pytest
from moirais.fn.sgtnlap import sgt_normalised_laplacian


def test_sgtnlap_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_normalised_laplacian(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtnlap_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_normalised_laplacian(A)
    assert isinstance(result, dict)
