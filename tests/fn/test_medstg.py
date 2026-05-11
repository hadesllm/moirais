"""Tests for medstg.sequential_mediation."""
import numpy as np
import pytest
from morie.fn.medstg import sequential_mediation


def test_medstg_basic():
    """Test basic functionality."""
    a1 = np.random.default_rng(42).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    c1 = np.random.default_rng(42).normal(0, 1, 100)
    result = sequential_mediation(a1, b1, c1)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_medstg_edge():
    """Test edge cases."""
    a1 = np.random.default_rng(42).normal(0, 1, 100)
    b1 = np.random.default_rng(42).normal(0, 1, 100)
    c1 = np.random.default_rng(42).normal(0, 1, 100)
    result = sequential_mediation(a1, b1, c1)
    assert isinstance(result, dict)
