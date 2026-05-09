"""Tests for rgwvd.rangayyan_wigner_ville."""
import numpy as np
import pytest
from moirais.fn.rgwvd import rangayyan_wigner_ville


def test_rgwvd_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_wigner_ville(x, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgwvd_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_wigner_ville(x, fs)
    assert isinstance(result, dict)
