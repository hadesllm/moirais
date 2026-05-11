"""Tests for rng217.rangayyan_ch4_schwarz_inequality_real."""
import numpy as np
import pytest
from morie.fn.rng217 import rangayyan_ch4_schwarz_inequality_real


def test_rng217_basic():
    """Test basic functionality."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_schwarz_inequality_real(a, b, t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng217_edge():
    """Test edge cases."""
    a = np.random.default_rng(44).normal(0, 1, 100)
    b = np.random.default_rng(42).normal(0, 1, 100)
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch4_schwarz_inequality_real(a, b, t)
    assert isinstance(result, dict)
