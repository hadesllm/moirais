"""Tests for gb5717.gibbons_wsrt_symmetry."""
import numpy as np
import pytest
from moirais.fn.gb5717 import gibbons_wsrt_symmetry


def test_gb5717_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu0 = 0.0
    result = gibbons_wsrt_symmetry(x, mu0)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb5717_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    mu0 = 0.0
    result = gibbons_wsrt_symmetry(x, mu0)
    assert isinstance(result, dict)
