"""Tests for egrch.egarch_model."""
import numpy as np
import pytest
from morie.fn.egrch import egarch_model


def test_egrch_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = egarch_model(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_egrch_edge():
    """Test edge cases."""
    result = egarch_model(np.array([42.0]))
    assert result['n'] == 1
