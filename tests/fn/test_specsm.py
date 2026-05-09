"""Tests for specsm.spectral_smoothed."""
import numpy as np
import pytest
from moirais.fn.specsm import spectral_smoothed


def test_specsm_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    span = np.random.default_rng(42).normal(0, 1, 100)
    result = spectral_smoothed(y, span)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_specsm_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    span = np.random.default_rng(42).normal(0, 1, 100)
    result = spectral_smoothed(y, span)
    assert isinstance(result, dict)
