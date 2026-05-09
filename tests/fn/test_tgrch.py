"""Tests for tgrch.tgarch_model."""
import numpy as np
import pytest
from moirais.fn.tgrch import tgarch_model


def test_tgrch_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = tgarch_model(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_tgrch_edge():
    """Test edge cases."""
    result = tgarch_model(np.array([42.0]))
    assert result['n'] == 1
