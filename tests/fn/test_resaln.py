"""Tests for resaln.resultant."""
import numpy as np
import pytest
from moirais.fn.resaln import resultant


def test_resaln_basic():
    """Test basic functionality."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = resultant(p, q)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_resaln_edge():
    """Test edge cases."""
    p = 5
    q = np.random.default_rng(42).normal(0, 1, 100)
    result = resultant(p, q)
    assert isinstance(result, dict)
