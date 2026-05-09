"""Tests for archm.arch_in_mean."""
import numpy as np
import pytest
from moirais.fn.archm import arch_in_mean


def test_archm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = arch_in_mean(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_archm_edge():
    """Test edge cases."""
    result = arch_in_mean(np.array([42.0]))
    assert result['n'] == 1
