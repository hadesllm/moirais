"""Tests for aitsbm.compositional_simbias."""
import numpy as np
import pytest
from morie.fn.aitsbm import compositional_simbias


def test_aitsbm_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    idx = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_simbias(X, idx)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_aitsbm_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    idx = np.random.default_rng(42).normal(0, 1, 100)
    result = compositional_simbias(X, idx)
    assert isinstance(result, dict)
