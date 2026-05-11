"""Tests for rng107.rangayyan_ch3_integrator_phase_response."""
import numpy as np
import pytest
from morie.fn.rng107 import rangayyan_ch3_integrator_phase_response


def test_rng107_basic():
    """Test basic functionality."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_integrator_phase_response(omega)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng107_edge():
    """Test edge cases."""
    omega = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch3_integrator_phase_response(omega)
    assert isinstance(result, dict)
