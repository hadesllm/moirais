"""Tests for spcrad.spectral_radius."""
import numpy as np
import pytest
from morie.fn.spcrad import spectral_radius


def test_spcrad_basic():
    """Test basic functionality."""
    G = np.eye(10)
    result = spectral_radius(G)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_spcrad_edge():
    """Test edge cases."""
    G = np.eye(10)
    result = spectral_radius(G)
    assert isinstance(result, dict)
