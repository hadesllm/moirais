"""Tests for mtsmtv.mts_mtr_combined."""
import numpy as np
import pytest
from morie.fn.mtsmtv import mts_mtr_combined


def test_mtsmtv_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = mts_mtr_combined(y, D, y_min, y_max)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_mtsmtv_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    y_min = 0
    y_max = 100
    result = mts_mtr_combined(y, D, y_min, y_max)
    assert isinstance(result, dict)
