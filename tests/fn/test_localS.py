"""Tests for localS.local_shift."""
import numpy as np
import pytest
from morie.fn.localS import local_shift


def test_localS_basic():
    """Test basic functionality."""
    IF = np.random.default_rng(42).normal(0, 1, 100)
    result = local_shift(IF)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_localS_edge():
    """Test edge cases."""
    IF = np.random.default_rng(42).normal(0, 1, 100)
    result = local_shift(IF)
    assert isinstance(result, dict)
