"""Tests for pcadm.pca_dimension_reduction."""
import numpy as np
import pytest
from moirais.fn.pcadm import pca_dimension_reduction


def test_pcadm_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = pca_dimension_reduction(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_pcadm_edge():
    """Test edge cases."""
    result = pca_dimension_reduction(np.array([42.0]))
    assert result['n'] == 1
