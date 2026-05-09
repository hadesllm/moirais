"""Tests for jordCD.jordan_canonical."""
import numpy as np
import pytest
from moirais.fn.jordCD import jordan_canonical


def test_jordCD_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = jordan_canonical(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_jordCD_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = jordan_canonical(A)
    assert isinstance(result, dict)
