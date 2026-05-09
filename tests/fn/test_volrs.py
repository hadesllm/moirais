"""Tests for volrs.vol_rogers_satchell."""
import numpy as np
import pytest
from moirais.fn.volrs import vol_rogers_satchell


def test_volrs_basic():
    """Test basic functionality."""
    o = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    l = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_rogers_satchell(o, h, l, c)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_volrs_edge():
    """Test edge cases."""
    o = np.random.default_rng(42).normal(0, 1, 100)
    h = 0.3
    l = np.random.default_rng(42).normal(0, 1, 100)
    c = np.random.default_rng(42).normal(0, 1, 100)
    result = vol_rogers_satchell(o, h, l, c)
    assert isinstance(result, dict)
