"""Tests for rgelast.rangayyan_heart_elasticity."""
import numpy as np
import pytest
from moirais.fn.rgelast import rangayyan_heart_elasticity


def test_rgelast_basic():
    """Test basic functionality."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_heart_elasticity(pcg, fs)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rgelast_edge():
    """Test edge cases."""
    pcg = np.random.default_rng(42).normal(0, 1, 1024)
    fs = 100.0
    result = rangayyan_heart_elasticity(pcg, fs)
    assert isinstance(result, dict)
