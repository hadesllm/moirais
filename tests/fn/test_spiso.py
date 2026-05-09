"""Tests for spiso.schabenberger_isotropy_condition."""
import numpy as np
import pytest
from moirais.fn.spiso import schabenberger_isotropy_condition


def test_spiso_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = schabenberger_isotropy_condition(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_spiso_edge():
    """Test edge cases."""
    result = schabenberger_isotropy_condition(np.array([42.0]))
    assert result['n'] == 1
