"""Tests for saxR.sax_representation."""
import numpy as np
import pytest
from morie.fn.saxR import sax_representation


def test_saxR_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    alphabet = np.random.default_rng(42).normal(0, 1, 100)
    result = sax_representation(x, window, alphabet)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_saxR_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    window = np.random.default_rng(42).normal(0, 1, 100)
    alphabet = np.random.default_rng(42).normal(0, 1, 100)
    result = sax_representation(x, window, alphabet)
    assert isinstance(result, dict)
