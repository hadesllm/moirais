"""Tests for vecmf.vecm_estimation."""
import numpy as np
import pytest
from moirais.fn.vecmf import vecm_estimation


def test_vecmf_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = vecm_estimation(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_vecmf_edge():
    """Test edge cases."""
    result = vecm_estimation(np.array([42.0]))
    assert result['n'] == 1
