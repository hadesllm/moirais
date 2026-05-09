"""Tests for ghcrt.ghosal_contraction_rate."""
import numpy as np
import pytest
from moirais.fn.ghcrt import ghosal_contraction_rate


def test_ghcrt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_contraction_rate(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghcrt_edge():
    """Test edge cases."""
    result = ghosal_contraction_rate(np.array([42.0]))
    assert result['n'] == 1
