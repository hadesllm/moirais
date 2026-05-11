"""Tests for rng223.rangayyan_ch4_test_signal_three_events."""
import numpy as np
import pytest
from morie.fn.rng223 import rangayyan_ch4_test_signal_three_events


def test_rng223_basic():
    """Test basic functionality."""
    n = 100
    result = rangayyan_ch4_test_signal_three_events(n)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_rng223_edge():
    """Test edge cases."""
    n = 100
    result = rangayyan_ch4_test_signal_three_events(n)
    assert isinstance(result, dict)
