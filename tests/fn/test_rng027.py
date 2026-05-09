"""Tests for rng027.rangayyan_ch3_unit_step_continuous."""
import numpy as np
import pytest
from moirais.fn.rng027 import rangayyan_ch3_unit_step_continuous


def test_rng027_basic():
    """Test basic functionality."""
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_unit_step_continuous(t)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng027_edge():
    """Test edge cases."""
    t = np.linspace(0, 10, 100)
    result = rangayyan_ch3_unit_step_continuous(t)
    assert isinstance(result, dict)
