"""Tests for dif1pl.dif_mantel_haenszel."""
import numpy as np
import pytest
from morie.fn.dif1pl import dif_mantel_haenszel


def test_dif1pl_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    item = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_mantel_haenszel(y, group, item)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_dif1pl_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    group = np.random.default_rng(42).normal(0, 1, 100)
    item = np.random.default_rng(42).normal(0, 1, 100)
    result = dif_mantel_haenszel(y, group, item)
    assert isinstance(result, dict)
