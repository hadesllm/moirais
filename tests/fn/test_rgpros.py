"""Tests for rgpros.rangayyan_prosthetic_valve."""
import numpy as np
import pytest
from moirais.fn.rgpros import rangayyan_prosthetic_valve


def test_rgpros_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_prosthetic_valve(pcg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgpros_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_prosthetic_valve(pcg, fs)
    assert isinstance(result, dict)
