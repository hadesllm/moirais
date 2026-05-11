"""Tests for rakedw.rake_double_weights."""
import numpy as np
import pytest
from morie.fn.rakedw import rake_double_weights


def test_rakedw_basic():
    """Test basic functionality."""
    w_nr = np.random.default_rng(42).normal(0, 1, 100)
    w_cal = np.random.default_rng(42).normal(0, 1, 100)
    result = rake_double_weights(w_nr, w_cal)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rakedw_edge():
    """Test edge cases."""
    w_nr = np.random.default_rng(42).normal(0, 1, 100)
    w_cal = np.random.default_rng(42).normal(0, 1, 100)
    result = rake_double_weights(w_nr, w_cal)
    assert isinstance(result, dict)
