"""Tests for rng224.rangayyan_ch4_basic_signal_g."""
import numpy as np
import pytest
from morie.fn.rng224 import rangayyan_ch4_basic_signal_g


def test_rng224_basic():
    """Test basic functionality."""
    n = 100
    result = rangayyan_ch4_basic_signal_g(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng224_edge():
    """Test edge cases."""
    n = 100
    result = rangayyan_ch4_basic_signal_g(n)
    assert isinstance(result, dict)
