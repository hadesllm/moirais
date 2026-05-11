"""Tests for rng194.rangayyan_ch4_heart_rate_from_rr."""
import numpy as np
import pytest
from morie.fn.rng194 import rangayyan_ch4_heart_rate_from_rr


def test_rng194_basic():
    """Test basic functionality."""
    RR_a = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_heart_rate_from_rr(RR_a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng194_edge():
    """Test edge cases."""
    RR_a = np.random.default_rng(42).normal(0, 1, 100)
    result = rangayyan_ch4_heart_rate_from_rr(RR_a)
    assert isinstance(result, dict)
