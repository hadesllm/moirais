"""Tests for laurnt.laurent_series."""
import numpy as np
import pytest
from moirais.fn.laurnt import laurent_series


def test_laurnt_basic():
    """Test basic functionality."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = laurent_series(f, c, order)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_laurnt_edge():
    """Test edge cases."""
    f = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    order = 4
    result = laurent_series(f, c, order)
    assert isinstance(result, dict)
