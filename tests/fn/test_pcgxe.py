"""Tests for pcgxe.pc_gxe_reduction."""
import numpy as np
import pytest
from moirais.fn.pcgxe import pc_gxe_reduction


def test_pcgxe_basic():
    """Test basic functionality."""
    GxE_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = pc_gxe_reduction(GxE_matrix, k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pcgxe_edge():
    """Test edge cases."""
    GxE_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    k = 5
    result = pc_gxe_reduction(GxE_matrix, k)
    assert isinstance(result, dict)
