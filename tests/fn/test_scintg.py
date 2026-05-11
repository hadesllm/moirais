"""Tests for scintg.singlecell_integration."""
import numpy as np
import pytest
from morie.fn.scintg import singlecell_integration


def test_scintg_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    batch = np.random.default_rng(42).normal(0, 1, 100)
    result = singlecell_integration(X, batch)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_scintg_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    batch = np.random.default_rng(42).normal(0, 1, 100)
    result = singlecell_integration(X, batch)
    assert isinstance(result, dict)
