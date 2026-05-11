"""Tests for sysmpl.systematic_sample."""
import numpy as np
import pytest
from morie.fn.sysmpl import systematic_sample


def test_sysmpl_basic():
    """Test basic functionality."""
    frame = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = systematic_sample(frame, n)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_sysmpl_edge():
    """Test edge cases."""
    frame = np.random.default_rng(42).normal(0, 1, 100)
    n = 100
    result = systematic_sample(frame, n)
    assert isinstance(result, dict)
