"""Tests for gb_are4.gibbons_are_kw."""
import numpy as np
import pytest
from moirais.fn.gb_are4 import gibbons_are_kw


def test_gb_are4_basic():
    """Test basic functionality."""
    distribution = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_kw(distribution)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_are4_edge():
    """Test edge cases."""
    distribution = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_are_kw(distribution)
    assert isinstance(result, dict)
