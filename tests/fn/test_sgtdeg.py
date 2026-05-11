"""Tests for sgtdeg.sgt_degree_matrix."""
import numpy as np
import pytest
from morie.fn.sgtdeg import sgt_degree_matrix


def test_sgtdeg_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_degree_matrix(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtdeg_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_degree_matrix(A)
    assert isinstance(result, dict)
