"""Tests for middle.middle_out."""
import numpy as np
import pytest
from moirais.fn.middle import middle_out


def test_middle_basic():
    """Test basic functionality."""
    middle = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = middle_out(middle, S)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_middle_edge():
    """Test edge cases."""
    middle = np.random.default_rng(42).normal(0, 1, 100)
    S = np.random.default_rng(42).normal(0, 1, 100)
    result = middle_out(middle, S)
    assert isinstance(result, dict)
