"""Tests for hadcrut.hadcrut."""
import numpy as np
import pytest
from morie.fn.hadcrut import hadcrut


def test_hadcrut_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    sst = np.random.default_rng(42).normal(0, 1, 100)
    result = hadcrut(T, sst)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_hadcrut_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    sst = np.random.default_rng(42).normal(0, 1, 100)
    result = hadcrut(T, sst)
    assert isinstance(result, dict)
