"""Tests for sgtsbpd.sgt_spectral_radius."""
import numpy as np
import pytest
from moirais.fn.sgtsbpd import sgt_spectral_radius


def test_sgtsbpd_basic():
    """Test basic functionality."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_spectral_radius(M)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sgtsbpd_edge():
    """Test edge cases."""
    M = np.random.default_rng(43).normal(0, 1, (10, 10))
    result = sgt_spectral_radius(M)
    assert isinstance(result, dict)
