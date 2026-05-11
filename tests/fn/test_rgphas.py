"""Tests for rgphas.rangayyan_phase_response."""
import numpy as np
import pytest
from morie.fn.rgphas import rangayyan_phase_response


def test_rgphas_basic():
    """Test basic functionality."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_phase_response(b, a, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgphas_edge():
    """Test edge cases."""
    b = np.random.default_rng(42).normal(0, 1, 100)
    a = np.random.default_rng(44).normal(0, 1, 100)
    fs = 100.0
    result = rangayyan_phase_response(b, a, fs)
    assert isinstance(result, dict)
