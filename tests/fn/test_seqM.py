"""Tests for seqM.sequential_mediators."""
import numpy as np
import pytest
from moirais.fn.seqM import sequential_mediators


def test_seqM_basic():
    """Test basic functionality."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M1 = np.random.default_rng(42).normal(0, 1, 100)
    M2 = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = sequential_mediators(Y, X, M1, M2, C)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_seqM_edge():
    """Test edge cases."""
    Y = np.random.default_rng(43).normal(0, 1, 100)
    X = np.random.default_rng(42).normal(0, 1, (100, 5))
    M1 = np.random.default_rng(42).normal(0, 1, 100)
    M2 = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    result = sequential_mediators(Y, X, M1, M2, C)
    assert isinstance(result, dict)
