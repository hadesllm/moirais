"""Tests for fzmrl.fauzi_mrl_asymptotic."""
import numpy as np
import pytest
from morie.fn.fzmrl import fauzi_mrl_asymptotic


def test_fzmrl_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = fauzi_mrl_asymptotic(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_fzmrl_edge():
    """Test edge cases."""
    result = fauzi_mrl_asymptotic(np.array([42.0]))
    assert result['n'] == 1
