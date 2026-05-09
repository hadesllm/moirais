"""Tests for optcl.optimal_classification."""
import numpy as np
import pytest
from moirais.fn.optcl import optimal_classification


def test_optcl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = optimal_classification(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_optcl_edge():
    """Test edge cases."""
    result = optimal_classification(np.array([42.0]))
    assert result['n'] == 1
