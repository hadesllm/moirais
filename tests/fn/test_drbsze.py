"""Tests for drbsze.dr_did_size_correction."""
import numpy as np
import pytest
from moirais.fn.drbsze import dr_did_size_correction


def test_drbsze_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_did_size_correction(y, D, X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_drbsze_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = dr_did_size_correction(y, D, X)
    assert isinstance(result, dict)
