"""Tests for rng090.rangayyan_ch3_hann_impulse_response."""
import numpy as np
import pytest
from moirais.fn.rng090 import rangayyan_ch3_hann_impulse_response


def test_rng090_basic():
    """Test basic functionality."""
    n = 100
    result = rangayyan_ch3_hann_impulse_response(n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng090_edge():
    """Test edge cases."""
    n = 100
    result = rangayyan_ch3_hann_impulse_response(n)
    assert isinstance(result, dict)
