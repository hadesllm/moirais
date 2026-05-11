"""Tests for dccmd.dcc_multivariate_garch."""
import numpy as np
import pytest
from morie.fn.dccmd import dcc_multivariate_garch


def test_dccmd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = dcc_multivariate_garch(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_dccmd_edge():
    """Test edge cases."""
    result = dcc_multivariate_garch(np.array([42.0]))
    assert result['n'] == 1
