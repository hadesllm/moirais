"""Tests for gpdsh.gp_density_shift."""
import numpy as np
import pytest
from moirais.fn.gpdsh import gp_density_shift


def test_gpdsh_basic():
    """Test basic functionality."""
    y_stream = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = gp_density_shift(y_stream, window, tau)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_gpdsh_edge():
    """Test edge cases."""
    y_stream = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    tau = 0.1
    result = gp_density_shift(y_stream, window, tau)
    assert isinstance(result, dict)
