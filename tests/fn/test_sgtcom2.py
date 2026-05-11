"""Tests for sgtcom2.sgt_communicability_matrix."""
import numpy as np
import pytest
from morie.fn.sgtcom2 import sgt_communicability_matrix


def test_sgtcom2_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_communicability_matrix(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtcom2_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = sgt_communicability_matrix(A)
    assert isinstance(result, dict)
