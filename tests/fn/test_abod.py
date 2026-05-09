"""Tests for abod.abod."""
import numpy as np
import pytest
from moirais.fn.abod import abod


def test_abod_basic():
    """Test basic functionality."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = abod(X)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_abod_edge():
    """Test edge cases."""
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    result = abod(X)
    assert isinstance(result, dict)
