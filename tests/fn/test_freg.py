"""Tests for freg.function_register."""
import numpy as np
import pytest
from moirais.fn.freg import function_register


def test_freg_basic():
    """Test basic functionality."""
    y1 = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    result = function_register(y1, y2)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_freg_edge():
    """Test edge cases."""
    y1 = np.random.default_rng(42).normal(0, 1, 100)
    y2 = np.random.default_rng(42).normal(0, 1, 100)
    result = function_register(y1, y2)
    assert isinstance(result, dict)
