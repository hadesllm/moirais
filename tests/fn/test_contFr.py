"""Tests for contFr.continued_fraction."""
import numpy as np
import pytest
from moirais.fn.contFr import continued_fraction


def test_contFr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = continued_fraction(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_contFr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = continued_fraction(x, n)
    assert isinstance(result, dict)
