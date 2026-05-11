"""Tests for lradw.lr_warmup."""
import numpy as np
import pytest
from morie.fn.lradw import lr_warmup


def test_lradw_basic():
    """Test basic functionality."""
    x = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
    result = lr_warmup(x)
    assert 'estimate' in result
    assert abs(result['estimate'] - 3.0) < 0.01


def test_lradw_edge():
    """Test edge cases."""
    result = lr_warmup(np.array([42.0]))
    assert result['n'] == 1
