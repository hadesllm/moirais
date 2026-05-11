"""Tests for rng136.rangayyan_ch3_notch_filter_60Hz."""
import numpy as np
import pytest
from morie.fn.rng136 import rangayyan_ch3_notch_filter_60Hz


def test_rng136_basic():
    """Test basic functionality."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_notch_filter_60Hz(z)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_rng136_edge():
    """Test edge cases."""
    z = np.random.default_rng(44).normal(0, 1, 100)
    result = rangayyan_ch3_notch_filter_60Hz(z)
    assert isinstance(result, dict)
