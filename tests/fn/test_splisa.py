"""Tests for splisa.schabenberger_lisa."""
import numpy as np
import pytest
from moirais.fn.splisa import schabenberger_lisa


def test_splisa_basic():
    """Test basic functionality."""
    x = np.arange(10, dtype=float)
    y = x * 2 + 1
    result = schabenberger_lisa(x, y)
    assert abs(result['statistic'] - 1.0) < 0.01


def test_splisa_edge():
    """Test edge cases."""
    result = schabenberger_lisa(np.array([1.0, 2.0]), np.array([3.0, 4.0]))
    assert result['n'] == 2
