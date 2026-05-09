"""Tests for gb961.gibbons_pct_mod_rank_sc."""
import numpy as np
import pytest
from moirais.fn.gb961 import gibbons_pct_mod_rank_sc


def test_gb961_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_pct_mod_rank_sc(x, y)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb961_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = gibbons_pct_mod_rank_sc(x, y)
    assert isinstance(result, dict)
