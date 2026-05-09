"""Tests for rejct.rejection_point."""
import numpy as np
import pytest
from moirais.fn.rejct import rejection_point


def test_rejct_basic():
    """Test basic functionality."""
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = rejection_point(psi)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rejct_edge():
    """Test edge cases."""
    psi = np.random.default_rng(42).normal(0, 1, 100)
    result = rejection_point(psi)
    assert isinstance(result, dict)
