"""Tests for tmpsc.temperature_scaling."""
import numpy as np
import pytest
from moirais.fn.tmpsc import temperature_scaling


def test_tmpsc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = temperature_scaling(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_tmpsc_edge():
    """Test edge cases."""
    result = temperature_scaling(np.array([42.0]))
    assert result['n'] == 1
