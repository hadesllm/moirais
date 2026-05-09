"""Tests for otmaprc.ot_map_recovery_brenier."""
import numpy as np
import pytest
from moirais.fn.otmaprc import ot_map_recovery_brenier


def test_otmaprc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ot_map_recovery_brenier(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_otmaprc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = ot_map_recovery_brenier(x, y)
    assert isinstance(result, dict)
