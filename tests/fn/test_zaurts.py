"""Tests for zaurts.zivot_andrews_unit_root."""
import numpy as np
import pytest
from morie.fn.zaurts import zivot_andrews_unit_root


def test_zaurts_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = zivot_andrews_unit_root(x, model, lags)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_zaurts_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    model = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = zivot_andrews_unit_root(x, model, lags)
    assert isinstance(result, dict)
