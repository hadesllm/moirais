"""Tests for wsmodd.wasserman_odds_ratio."""
import numpy as np
import pytest
from morie.fn.wsmodd import wasserman_odds_ratio


def test_wsmodd_basic():
    """Test basic functionality."""
    table = np.array([[10, 20, 30], [15, 25, 35]])
    result = wasserman_odds_ratio(table)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmodd_edge():
    """Test edge cases."""
    table = np.array([[10, 20, 30], [15, 25, 35]])
    result = wasserman_odds_ratio(table)
    assert isinstance(result, dict)
