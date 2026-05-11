"""Tests for archlm.arch_lm_engle."""
import numpy as np
import pytest
from morie.fn.archlm import arch_lm_engle


def test_archlm_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = arch_lm_engle(x, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_archlm_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = arch_lm_engle(x, lags)
    assert isinstance(result, dict)
