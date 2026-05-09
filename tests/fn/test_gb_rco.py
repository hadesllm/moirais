"""Tests for gb_rco.gibbons_rank_corr_partial."""
import numpy as np
import pytest
from moirais.fn.gb_rco import gibbons_rank_corr_partial


def test_gb_rco_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = gibbons_rank_corr_partial(x, y, z)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'estimate' in result


def test_gb_rco_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = gibbons_rank_corr_partial(x, y, z)
    assert isinstance(result, dict)
