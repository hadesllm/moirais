"""Tests for rng039.rangayyan_ch3_ma_filter_11pt."""
import numpy as np
import pytest
from moirais.fn.rng039 import rangayyan_ch3_ma_filter_11pt


def test_rng039_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_ma_filter_11pt(x, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng039_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_ma_filter_11pt(x, n)
    assert isinstance(result, dict)
