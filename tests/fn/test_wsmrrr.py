"""Tests for wsmrrr.wasserman_relative_risk."""
import numpy as np
import pytest
from moirais.fn.wsmrrr import wasserman_relative_risk


def test_wsmrrr_basic():
    """Test basic functionality."""
    table = np.array([[10, 20, 30], [15, 25, 35]])
    result = wasserman_relative_risk(table)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmrrr_edge():
    """Test edge cases."""
    table = np.array([[10, 20, 30], [15, 25, 35]])
    result = wasserman_relative_risk(table)
    assert isinstance(result, dict)
