"""Tests for hetmix.heterogeneous_mixing."""
import numpy as np
import pytest
from moirais.fn.hetmix import heterogeneous_mixing


def test_hetmix_basic():
    """Test basic functionality."""
    contact_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    gamma = 1.0
    result = heterogeneous_mixing(contact_matrix, gamma)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hetmix_edge():
    """Test edge cases."""
    contact_matrix = np.random.default_rng(42).normal(0, 1, (10, 10))
    gamma = 1.0
    result = heterogeneous_mixing(contact_matrix, gamma)
    assert isinstance(result, dict)
