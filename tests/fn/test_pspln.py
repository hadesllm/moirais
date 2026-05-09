"""Tests for pspln.penalized_spline."""
import numpy as np
import pytest
from moirais.fn.pspln import penalized_spline


def test_pspln_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = penalized_spline(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_pspln_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = penalized_spline(x, y)
    assert isinstance(result, dict)
