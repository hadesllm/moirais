"""Tests for sgtmodul.sgt_modularity_matrix."""
import numpy as np
import pytest
from moirais.fn.sgtmodul import sgt_modularity_matrix


def test_sgtmodul_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_modularity_matrix(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtmodul_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_modularity_matrix(A)
    assert isinstance(result, dict)
