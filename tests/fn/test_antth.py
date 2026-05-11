"""Tests for antth.antithetic_variates."""
import numpy as np
import pytest
from morie.fn.antth import antithetic_variates


def test_antth_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = antithetic_variates(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_antth_edge():
    """Test edge cases."""
    result = antithetic_variates(np.array([42.0]))
    assert result['n'] == 1
