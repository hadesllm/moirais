"""Tests for rgsam.rangayyan_sample_entropy."""
import numpy as np
import pytest
from moirais.fn.rgsam import rangayyan_sample_entropy


def test_rgsam_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = rangayyan_sample_entropy(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_rgsam_edge():
    """Test edge cases."""
    result = rangayyan_sample_entropy(np.array([42.0]))
    assert result['n'] == 1
