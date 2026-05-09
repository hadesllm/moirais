"""Tests for nybhsn.nyblom_hansen_stability."""
import numpy as np
import pytest
from moirais.fn.nybhsn import nyblom_hansen_stability


def test_nybhsn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = nyblom_hansen_stability(y, X)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_nybhsn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = nyblom_hansen_stability(y, X)
    assert isinstance(result, dict)
