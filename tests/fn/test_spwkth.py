"""Tests for spwkth.schabenberger_wiener_khinchin."""
import numpy as np
import pytest
from moirais.fn.spwkth import schabenberger_wiener_khinchin


def test_spwkth_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = schabenberger_wiener_khinchin(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_spwkth_edge():
    """Test edge cases."""
    result = schabenberger_wiener_khinchin(np.array([42.0]))
    assert result['n'] == 1
