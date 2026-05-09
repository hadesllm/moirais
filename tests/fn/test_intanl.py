"""Tests for intanl.interaction_analysis."""
import numpy as np
import pytest
from moirais.fn.intanl import interaction_analysis


def test_intanl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = interaction_analysis(y, A, V, H)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_intanl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    V = np.random.default_rng(42).normal(0, 1, 100)
    H = np.random.default_rng(42).normal(0, 1, 100)
    result = interaction_analysis(y, A, V, H)
    assert isinstance(result, dict)
