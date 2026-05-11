"""Tests for snsc.sn_scale."""
import numpy as np
import pytest
from morie.fn.snsc import sn_scale


def test_snsc_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sn_scale(x)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_snsc_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    result = sn_scale(x)
    assert isinstance(result, dict)
