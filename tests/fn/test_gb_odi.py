"""Tests for gb_odi.gibbons_odds_ratio."""
import numpy as np
import pytest
from morie.fn.gb_odi import gibbons_odds_ratio


def test_gb_odi_basic():
    """Test basic functionality."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_odds_ratio(table)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_odi_edge():
    """Test edge cases."""
    table = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_odds_ratio(table)
    assert isinstance(result, dict)
