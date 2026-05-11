"""Tests for ssmod.state_space_model."""
import numpy as np
import pytest
from morie.fn.ssmod import state_space_model


def test_ssmod_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = state_space_model(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_ssmod_edge():
    """Test edge cases."""
    result = state_space_model(np.array([42.0]))
    assert result['n'] == 1
