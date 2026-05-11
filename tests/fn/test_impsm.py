"""Tests for impsm.importance_sampling."""
import numpy as np
import pytest
from morie.fn.impsm import importance_sampling


def test_impsm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = importance_sampling(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_impsm_edge():
    """Test edge cases."""
    result = importance_sampling(np.array([42.0]))
    assert result['n'] == 1
