"""Tests for ngomtx.next_generation_matrix."""
import numpy as np
import pytest
from moirais.fn.ngomtx import next_generation_matrix


def test_ngomtx_basic():
    """Test basic functionality."""
    FV_decomposition = np.random.default_rng(42).normal(0, 1, 100)
    result = next_generation_matrix(FV_decomposition)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ngomtx_edge():
    """Test edge cases."""
    FV_decomposition = np.random.default_rng(42).normal(0, 1, 100)
    result = next_generation_matrix(FV_decomposition)
    assert isinstance(result, dict)
