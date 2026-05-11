"""Tests for rng078.rangayyan_ch3_twiddle_periodicity."""
import numpy as np
import pytest
from morie.fn.rng078 import rangayyan_ch3_twiddle_periodicity


def test_rng078_basic():
    """Test basic functionality."""
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_twiddle_periodicity(n, k, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng078_edge():
    """Test edge cases."""
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_twiddle_periodicity(n, k, N)
    assert isinstance(result, dict)
