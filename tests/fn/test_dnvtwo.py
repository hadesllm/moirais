"""Tests for dnvtwo.dinov2_repr."""
import numpy as np
import pytest
from morie.fn.dnvtwo import dinov2_repr


def test_dnvtwo_basic():
    """Test basic functionality."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = dinov2_repr(x, student, teacher)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_dnvtwo_edge():
    """Test edge cases."""
    x = np.random.default_rng(42).normal(0, 1, 100)
    student = np.random.default_rng(42).normal(0, 1, 100)
    teacher = np.random.default_rng(42).normal(0, 1, 100)
    result = dinov2_repr(x, student, teacher)
    assert isinstance(result, dict)
