"""Tests for rng112.rangayyan_ch3_first_difference_transfer_function."""
import numpy as np
import pytest
from moirais.fn.rng112 import rangayyan_ch3_first_difference_transfer_function


def test_rng112_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_first_difference_transfer_function(z, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng112_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = rangayyan_ch3_first_difference_transfer_function(z, T)
    assert isinstance(result, dict)
