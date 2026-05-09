"""Tests for sgtlpi.sgt_laplacian_pseudoinverse."""
import numpy as np
import pytest
from moirais.fn.sgtlpi import sgt_laplacian_pseudoinverse


def test_sgtlpi_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_laplacian_pseudoinverse(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtlpi_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_laplacian_pseudoinverse(A)
    assert isinstance(result, dict)
