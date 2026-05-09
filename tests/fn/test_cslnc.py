"""Tests for cslnc.cosine_lr_schedule."""
import numpy as np
import pytest
from moirais.fn.cslnc import cosine_lr_schedule


def test_cslnc_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = cosine_lr_schedule(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_cslnc_edge():
    """Test edge cases."""
    result = cosine_lr_schedule(np.array([42.0]))
    assert result['n'] == 1
