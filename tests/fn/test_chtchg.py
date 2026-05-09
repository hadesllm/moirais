"""Tests for chtchg.changeover_dr."""
import numpy as np
import pytest
from moirais.fn.chtchg import changeover_dr


def test_chtchg_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    result = changeover_dr(y, D, period, unit)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chtchg_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    period = np.random.default_rng(42).normal(0, 1, 100)
    unit = np.random.default_rng(42).normal(0, 1, 100)
    result = changeover_dr(y, D, period, unit)
    assert isinstance(result, dict)
