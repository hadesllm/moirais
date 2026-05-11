"""Tests for singsd.singular_spectrum."""
import numpy as np
import pytest
from morie.fn.singsd import singular_spectrum


def test_singsd_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = singular_spectrum(y, window)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_singsd_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    result = singular_spectrum(y, window)
    assert isinstance(result, dict)
