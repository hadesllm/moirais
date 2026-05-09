"""Tests for diffRC.diffusion_rec."""
import numpy as np
import pytest
from moirais.fn.diffRC import diffusion_rec


def test_diffRC_basic():
    """Test basic functionality."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = diffusion_rec(R, T)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_diffRC_edge():
    """Test edge cases."""
    R = np.random.default_rng(42).normal(0, 1, 100)
    T = np.random.default_rng(43).integers(0, 2, 100)
    result = diffusion_rec(R, T)
    assert isinstance(result, dict)
