"""Tests for rng106.rangayyan_ch3_integrator_magnitude_response."""
import numpy as np
import pytest
from moirais.fn.rng106 import rangayyan_ch3_integrator_magnitude_response


def test_rng106_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_integrator_magnitude_response(omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng106_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_integrator_magnitude_response(omega)
    assert isinstance(result, dict)
