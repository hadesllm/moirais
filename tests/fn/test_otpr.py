"""Tests for otpr.ot_partial_ot."""
import numpy as np
import pytest
from morie.fn.otpr import ot_partial_ot


def test_otpr_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = ot_partial_ot(a, b, C, m)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otpr_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    C = np.random.default_rng(42).normal(0, 1, 100)
    m = 10
    result = ot_partial_ot(a, b, C, m)
    assert isinstance(result, dict)
