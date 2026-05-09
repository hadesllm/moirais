"""Tests for diaspath.diameter."""
import numpy as np
import pytest
from moirais.fn.diaspath import diameter


def test_diaspath_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = diameter(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diaspath_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = diameter(G)
    assert isinstance(result, dict)
