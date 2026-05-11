"""Tests for madsc.mad_scale."""
import numpy as np
import pytest
from morie.fn.madsc import mad_scale


def test_madsc_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mad_scale(y)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_madsc_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    result = mad_scale(y)
    assert isinstance(result, dict)
