"""Tests for propc.prophet_components."""
import numpy as np
import pytest
from moirais.fn.propc import prophet_components


def test_propc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = prophet_components(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_propc_edge():
    """Test edge cases."""
    result = prophet_components(np.array([42.0]))
    assert result['n'] == 1
