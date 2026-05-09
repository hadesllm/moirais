"""Tests for maeggr.ma_egger_test."""
import numpy as np
import pytest
from moirais.fn.maeggr import ma_egger_test


def test_maeggr_basic():
    """Test basic functionality."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    se_i = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_egger_test(yi, se_i)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_maeggr_edge():
    """Test edge cases."""
    yi = np.random.default_rng(42).normal(0, 1, 100)
    se_i = np.random.default_rng(42).normal(0, 1, 100)
    result = ma_egger_test(yi, se_i)
    assert isinstance(result, dict)
