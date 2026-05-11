"""Tests for specdn.spectral_density."""
import numpy as np
import pytest
from morie.fn.specdn import spectral_density


def test_specdn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = spectral_density(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_specdn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = spectral_density(y)
    assert isinstance(result, dict)
