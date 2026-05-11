"""Tests for bdintp.bound_intersection."""
import numpy as np
import pytest
from morie.fn.bdintp import bound_intersection


def test_bdintp_basic():
    """Test basic functionality."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_intersection(y, D, moments)
    assert isinstance(result, dict)
    assert 'estimate' in result or 'statistic' in result


def test_bdintp_edge():
    """Test edge cases."""
    y = np.random.default_rng(43).normal(0, 1, 100)
    D = np.random.default_rng(42).normal(0, 1, 100)
    moments = np.random.default_rng(42).normal(0, 1, 100)
    result = bound_intersection(y, D, moments)
    assert isinstance(result, dict)
