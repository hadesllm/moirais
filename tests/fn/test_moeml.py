"""Tests for moeml.mixture_of_experts."""
import numpy as np
import pytest
from moirais.fn.moeml import mixture_of_experts


def test_moeml_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = mixture_of_experts(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_moeml_edge():
    """Test edge cases."""
    result = mixture_of_experts(np.array([42.0]))
    assert result['n'] == 1
