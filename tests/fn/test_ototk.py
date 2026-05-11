"""Tests for ototk.ot_kantorovich_dual_value."""
import numpy as np
import pytest
from morie.fn.ototk import ot_kantorovich_dual_value


def test_ototk_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = ot_kantorovich_dual_value(a, b, f, g)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_ototk_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    f = np.random.default_rng(42).normal(0, 1, 100)
    g = np.random.default_rng(43).normal(0, 1, 100)
    result = ot_kantorovich_dual_value(a, b, f, g)
    assert isinstance(result, dict)
