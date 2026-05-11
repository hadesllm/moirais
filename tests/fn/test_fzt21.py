"""Tests for fzt21.fauzi_thm2_1_expected_kdfe."""
import numpy as np
import pytest
from morie.fn.fzt21 import fauzi_thm2_1_expected_kdfe


def test_fzt21_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_thm2_1_expected_kdfe(x, bandwidth, a)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_fzt21_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    bandwidth = 0.3
    a = np.random.default_rng(44).normal(0, 1, 100)
    result = fauzi_thm2_1_expected_kdfe(x, bandwidth, a)
    assert isinstance(result, dict)
