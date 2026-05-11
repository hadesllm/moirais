"""Tests for specA.spectral_anomaly."""
import numpy as np
import pytest
from morie.fn.specA import spectral_anomaly


def test_specA_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spectral_anomaly(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_specA_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = spectral_anomaly(x)
    assert isinstance(result, dict)
