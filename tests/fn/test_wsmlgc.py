"""Tests for wsmlgc.wasserman_log_linear."""
import numpy as np
import pytest
from moirais.fn.wsmlgc import wasserman_log_linear


def test_wsmlgc_basic():
    """Test basic functionality."""
    table = np.array([[10, 20, 30], [15, 25, 35]])
    result = wasserman_log_linear(table)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmlgc_edge():
    """Test edge cases."""
    table = np.array([[10, 20, 30], [15, 25, 35]])
    result = wasserman_log_linear(table)
    assert isinstance(result, dict)
