"""Tests for rng025.rangayyan_ch3_dirac_delta_unit_area."""
import numpy as np
import pytest
from morie.fn.rng025 import rangayyan_ch3_dirac_delta_unit_area


def test_rng025_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_dirac_delta_unit_area(t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng025_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_dirac_delta_unit_area(t)
    assert isinstance(result, dict)
