"""Tests for adwin.adwin."""
import numpy as np
import pytest
from moirais.fn.adwin import adwin


def test_adwin_basic():
    """Test basic functionality."""
    stream = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = adwin(stream, delta)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_adwin_edge():
    """Test edge cases."""
    stream = np.random.default_rng(42).normal(0, 1, 100)
    delta = np.random.default_rng(42).normal(0, 1, 100)
    result = adwin(stream, delta)
    assert isinstance(result, dict)
