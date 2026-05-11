"""Tests for sptag.spatial_agreement."""
import numpy as np
import pytest
from morie.fn.sptag import spatial_agreement


def test_sptag_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = spatial_agreement(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_sptag_edge():
    """Test edge cases."""
    result = spatial_agreement(np.array([42.0]))
    assert result['n'] == 1
