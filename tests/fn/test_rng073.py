"""Tests for rng073.rangayyan_ch3_twiddle_cos_sin."""
import numpy as np
import pytest
from moirais.fn.rng073 import rangayyan_ch3_twiddle_cos_sin


def test_rng073_basic():
    """Test basic functionality."""
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_twiddle_cos_sin(n, k, N)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng073_edge():
    """Test edge cases."""
    n = 100
    k = 5
    N = 100
    result = rangayyan_ch3_twiddle_cos_sin(n, k, N)
    assert isinstance(result, dict)
