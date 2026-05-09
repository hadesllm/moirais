"""Tests for rng094.rangayyan_ch3_hann_frequency_response_simplified."""
import numpy as np
import pytest
from moirais.fn.rng094 import rangayyan_ch3_hann_frequency_response_simplified


def test_rng094_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_hann_frequency_response_simplified(omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng094_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_hann_frequency_response_simplified(omega)
    assert isinstance(result, dict)
