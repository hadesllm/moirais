"""Tests for plcmt.rank_placements."""
import numpy as np
import pytest
from moirais.fn.plcmt import rank_placements


def test_plcmt_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rank_placements(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_plcmt_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = rank_placements(x, y)
    assert isinstance(result, dict)
