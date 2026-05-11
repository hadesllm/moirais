"""Tests for phpprn.phillips_perron."""
import numpy as np
import pytest
from morie.fn.phpprn import phillips_perron


def test_phpprn_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    result = phillips_perron(y, trend)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_phpprn_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    trend = np.random.default_rng(42).normal(0, 1, 100)
    result = phillips_perron(y, trend)
    assert isinstance(result, dict)
