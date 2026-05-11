"""Tests for mdrnk.midranks."""
import numpy as np
import pytest
from morie.fn.mdrnk import midranks


def test_mdrnk_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = midranks(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_mdrnk_edge():
    """Test edge cases."""
    result = midranks(np.array([42.0]))
    assert result['n'] == 1
