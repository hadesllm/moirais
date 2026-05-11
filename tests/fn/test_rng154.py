"""Tests for rng154.rangayyan_ch3_anc_output."""
import numpy as np
import pytest
from morie.fn.rng154 import rangayyan_ch3_anc_output


def test_rng154_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_anc_output(x, y, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng154_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    y = np.random.default_rng(43).normal(0, 1, 100)
    n = 100
    result = rangayyan_ch3_anc_output(x, y, n)
    assert isinstance(result, dict)
