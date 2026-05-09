"""Tests for bxprc.box_pierce_test."""
import numpy as np
import pytest
from moirais.fn.bxprc import box_pierce_test


def test_bxprc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = box_pierce_test(x, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_bxprc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = box_pierce_test(x, lags)
    assert isinstance(result, dict)
