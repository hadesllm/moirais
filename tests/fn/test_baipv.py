"""Tests for baipv.bai_perron_multiple_breaks."""
import numpy as np
import pytest
from morie.fn.baipv import bai_perron_multiple_breaks


def test_baipv_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_breaks = np.random.default_rng(42).normal(0, 1, 100)
    result = bai_perron_multiple_breaks(x, max_breaks)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_baipv_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    max_breaks = np.random.default_rng(42).normal(0, 1, 100)
    result = bai_perron_multiple_breaks(x, max_breaks)
    assert isinstance(result, dict)
