"""Tests for polrz.polarization_index."""
import numpy as np
import pytest
from moirais.fn.polrz import polarization_index


def test_polrz_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = polarization_index(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_polrz_edge():
    """Test edge cases."""
    result = polarization_index(np.array([42.0]))
    assert result['n'] == 1
