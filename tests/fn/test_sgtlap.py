"""Tests for sgtlap.sgt_laplacian."""
import numpy as np
import pytest
from morie.fn.sgtlap import sgt_laplacian


def test_sgtlap_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_laplacian(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtlap_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_laplacian(A)
    assert isinstance(result, dict)
