"""Tests for mnpbt.multinomial_probit_spatial."""
import numpy as np
import pytest
from morie.fn.mnpbt import multinomial_probit_spatial


def test_mnpbt_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = multinomial_probit_spatial(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mnpbt_edge():
    """Test edge cases."""
    result = multinomial_probit_spatial(np.array([42.0]))
    assert result['n'] == 1
