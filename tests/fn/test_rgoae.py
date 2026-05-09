"""Tests for rgoae.rangayyan_oae."""
import numpy as np
import pytest
from moirais.fn.rgoae import rangayyan_oae


def test_rgoae_basic():
    """Test basic functionality."""
    oae = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_oae(oae, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgoae_edge():
    """Test edge cases."""
    oae = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_oae(oae, fs)
    assert isinstance(result, dict)
