"""Tests for rng100.rangayyan_ch3_ma_8point_frequency_response."""
import numpy as np
import pytest
from moirais.fn.rng100 import rangayyan_ch3_ma_8point_frequency_response


def test_rng100_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_ma_8point_frequency_response(omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng100_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_ma_8point_frequency_response(omega)
    assert isinstance(result, dict)
