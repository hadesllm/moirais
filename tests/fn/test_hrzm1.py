"""Tests for hrzm1.horowitz_mixture_model."""
import numpy as np
import pytest
from moirais.fn.hrzm1 import horowitz_mixture_model


def test_hrzm1_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = horowitz_mixture_model(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_hrzm1_edge():
    """Test edge cases."""
    result = horowitz_mixture_model(np.array([42.0]))
    assert result['n'] == 1
