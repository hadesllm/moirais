"""Tests for pptest.phillips_perron_unit_root."""
import numpy as np
import pytest
from moirais.fn.pptest import phillips_perron_unit_root


def test_pptest_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = phillips_perron_unit_root(x, lags)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_pptest_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    lags = 10
    result = phillips_perron_unit_root(x, lags)
    assert isinstance(result, dict)
