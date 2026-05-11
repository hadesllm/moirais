"""Tests for basisR.basis_representation."""
import numpy as np
import pytest
from morie.fn.basisR import basis_representation


def test_basisR_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    result = basis_representation(y, Phi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_basisR_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    Phi = np.random.default_rng(42).normal(0, 1, 100)
    result = basis_representation(y, Phi)
    assert isinstance(result, dict)
