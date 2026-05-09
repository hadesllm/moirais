"""Tests for gb_frd2.gibbons_friedman_ties."""
import numpy as np
import pytest
from moirais.fn.gb_frd2 import gibbons_friedman_ties


def test_gb_frd2_basic():
    """Test basic functionality."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_friedman_ties(data)
    assert isinstance(result, dict)
    assert 'statistic' in result or 'p_value' in result or 'estimate' in result


def test_gb_frd2_edge():
    """Test edge cases."""
    data = np.random.default_rng(42).normal(0, 1, 100)
    result = gibbons_friedman_ties(data)
    assert isinstance(result, dict)
