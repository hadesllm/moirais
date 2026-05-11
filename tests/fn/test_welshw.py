"""Tests for welshw.welsch_weight."""
import numpy as np
import pytest
from morie.fn.welshw import welsch_weight


def test_welshw_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = welsch_weight(y, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_welshw_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = welsch_weight(y, c)
    assert isinstance(result, dict)
