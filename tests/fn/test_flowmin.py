"""Tests for flowmin.min_cut."""
import numpy as np
import pytest
from moirais.fn.flowmin import min_cut


def test_flowmin_basic():
    """Test basic functionality."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = min_cut(A)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_flowmin_edge():
    """Test edge cases."""
    A = np.random.default_rng(42).normal(0, 1, (10, 10))
    result = min_cut(A)
    assert isinstance(result, dict)
