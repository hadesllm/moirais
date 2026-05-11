"""Tests for rgvocal.rangayyan_vocal_tract."""
import numpy as np
import pytest
from morie.fn.rgvocal import rangayyan_vocal_tract


def test_rgvocal_basic():
    """Test basic functionality."""
    speech = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_coeff = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_vocal_tract(speech, fs, n_coeff)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgvocal_edge():
    """Test edge cases."""
    speech = np.random.default_rng(42).normal(0, 1, 100)
    fs = 100.0
    n_coeff = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_vocal_tract(speech, fs, n_coeff)
    assert isinstance(result, dict)
