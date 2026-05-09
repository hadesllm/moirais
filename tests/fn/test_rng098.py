"""Tests for rng098.rangayyan_ch3_ma_8point_impulse_response."""
import numpy as np
import pytest
from moirais.fn.rng098 import rangayyan_ch3_ma_8point_impulse_response


def test_rng098_basic():
    """Test basic functionality."""
    n = 100
    result = rangayyan_ch3_ma_8point_impulse_response(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng098_edge():
    """Test edge cases."""
    n = 100
    result = rangayyan_ch3_ma_8point_impulse_response(n)
    assert isinstance(result, dict)
