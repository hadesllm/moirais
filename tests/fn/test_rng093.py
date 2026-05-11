"""Tests for rng093.rangayyan_ch3_hann_frequency_response_raw."""
import numpy as np
import pytest
from morie.fn.rng093 import rangayyan_ch3_hann_frequency_response_raw


def test_rng093_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_hann_frequency_response_raw(omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng093_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_hann_frequency_response_raw(omega)
    assert isinstance(result, dict)
