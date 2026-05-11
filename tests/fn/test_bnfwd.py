"""Tests for bnfwd.batch_norm_forward."""
import numpy as np
import pytest
from morie.fn.bnfwd import batch_norm_forward


def test_bnfwd_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = batch_norm_forward(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_bnfwd_edge():
    """Test edge cases."""
    result = batch_norm_forward(np.array([42.0]))
    assert result['n'] == 1
