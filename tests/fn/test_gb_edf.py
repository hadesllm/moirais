"""Tests for gb_edf.gibbons_edf_def."""
import numpy as np
import pytest
from morie.fn.gb_edf import gibbons_edf_def


def test_gb_edf_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_edf_def(x, data)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_edf_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_edf_def(x, data)
    assert isinstance(result, dict)
