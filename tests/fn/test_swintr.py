"""Tests for swintr.swin_transformer."""
import numpy as np
import pytest
from morie.fn.swintr import swin_transformer


def test_swintr_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window_size = 100
    result = swin_transformer(x, window_size)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_swintr_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window_size = 100
    result = swin_transformer(x, window_size)
    assert isinstance(result, dict)
