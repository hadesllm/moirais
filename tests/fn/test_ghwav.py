"""Tests for ghwav.ghosal_wavelet_prior."""
import numpy as np
import pytest
from moirais.fn.ghwav import ghosal_wavelet_prior


def test_ghwav_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = ghosal_wavelet_prior(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ghwav_edge():
    """Test edge cases."""
    result = ghosal_wavelet_prior(np.array([42.0]))
    assert result['n'] == 1
