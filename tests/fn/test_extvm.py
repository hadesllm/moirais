"""Tests for extvm.extreme_value_gev."""
import numpy as np
import pytest
from moirais.fn.extvm import extreme_value_gev


def test_extvm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = extreme_value_gev(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_extvm_edge():
    """Test edge cases."""
    result = extreme_value_gev(np.array([42.0]))
    assert result['n'] == 1
