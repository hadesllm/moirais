"""Tests for densty.density."""
import numpy as np
import pytest
from moirais.fn.densty import density


def test_densty_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = density(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_densty_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = density(G)
    assert isinstance(result, dict)
