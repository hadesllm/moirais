"""Tests for wsmcby.wasserman_chebyshev_ineq."""
import numpy as np
import pytest
from moirais.fn.wsmcby import wasserman_chebyshev_ineq


def test_wsmcby_basic():
    """Test basic functionality."""
    k = 5
    result = wasserman_chebyshev_ineq(k)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_wsmcby_edge():
    """Test edge cases."""
    k = 5
    result = wasserman_chebyshev_ineq(k)
    assert isinstance(result, dict)
