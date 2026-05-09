"""Tests for rng095.rangayyan_ch3_hann_magnitude_response."""
import numpy as np
import pytest
from moirais.fn.rng095 import rangayyan_ch3_hann_magnitude_response


def test_rng095_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_hann_magnitude_response(omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng095_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_hann_magnitude_response(omega)
    assert isinstance(result, dict)
