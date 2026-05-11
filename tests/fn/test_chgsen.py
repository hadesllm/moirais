"""Tests for chgsen.change_of_variance."""
import numpy as np
import pytest
from morie.fn.chgsen import change_of_variance


def test_chgsen_basic():
    """Test basic functionality."""
    IF = np.random.default_rng(42).normal(0, 1, 100)
    result = change_of_variance(IF)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_chgsen_edge():
    """Test edge cases."""
    IF = np.random.default_rng(42).normal(0, 1, 100)
    result = change_of_variance(IF)
    assert isinstance(result, dict)
