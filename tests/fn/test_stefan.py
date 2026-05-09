"""Tests for stefan.stefan_boltzmann."""
import numpy as np
import pytest
from moirais.fn.stefan import stefan_boltzmann


def test_stefan_basic():
    """Test basic functionality."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    emissivity = np.random.default_rng(42).normal(0, 1, 100)
    result = stefan_boltzmann(T, emissivity)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_stefan_edge():
    """Test edge cases."""
    T = np.random.default_rng(43).integers(0, 2, 100)
    emissivity = np.random.default_rng(42).normal(0, 1, 100)
    result = stefan_boltzmann(T, emissivity)
    assert isinstance(result, dict)
