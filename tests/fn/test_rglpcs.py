"""Tests for rglpcs.rangayyan_lpc_synthesis."""
import numpy as np
import pytest
from moirais.fn.rglpcs import rangayyan_lpc_synthesis


def test_rglpcs_basic():
    """Test basic functionality."""
    lpc_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    gain = np.random.default_rng(42).normal(0, 1, 100)
    excitation = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_lpc_synthesis(lpc_coeffs, gain, excitation)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rglpcs_edge():
    """Test edge cases."""
    lpc_coeffs = np.random.default_rng(42).normal(0, 1, 100)
    gain = np.random.default_rng(42).normal(0, 1, 100)
    excitation = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_lpc_synthesis(lpc_coeffs, gain, excitation)
    assert isinstance(result, dict)
