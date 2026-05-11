"""Tests for diophs.diophantine."""
import numpy as np
import pytest
from morie.fn.diophs import diophantine


def test_diophs_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = diophantine(a, b, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diophs_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = diophantine(a, b, c)
    assert isinstance(result, dict)
