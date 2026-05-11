"""Tests for lemR.leiden_grph."""
import numpy as np
import pytest
from morie.fn.lemR import leiden_grph


def test_lemR_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = leiden_grph(A, resolution)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_lemR_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    resolution = np.random.default_rng(42).normal(0, 1, 100)
    result = leiden_grph(A, resolution)
    assert isinstance(result, dict)
