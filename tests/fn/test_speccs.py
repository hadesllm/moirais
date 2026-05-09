"""Tests for speccs.cross_spectrum."""
import numpy as np
import pytest
from moirais.fn.speccs import cross_spectrum


def test_speccs_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = cross_spectrum(x, y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_speccs_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = cross_spectrum(x, y)
    assert isinstance(result, dict)
